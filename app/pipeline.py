import asyncio
import time
import json
import re
from fastapi import WebSocket
from app.config import settings
from app.vad import LocalVAD
from app.stt_engine import LocalSTT
from app.llm_client import LocalLLM
from app.tts_engine import LocalTTS
from app.tools import book_appointment

SYSTEM_PROMPT = """You are an operational dispatcher for City Health Diagnostics, Hyderabad.
Your sole job is appointment scheduling. Be crisp and conversational.
Keep spoken replies strictly under 8 words. No markdown formatting.

If the user provides a name, date, and time slot, output an action command:
ACTION: {"tool": "book_appointment", "args": {"caller_name": "...", "phone_number": "...", "date": "...", "time_slot": "..."}}
Follow the action command immediately with a spoken confirmation under 6 words.

Do not use ACTION for anything else. Never output ACTION for other requests.
User: "What is two plus two?"
Assistant: Two plus two is four.
User: "Please count slowly from one to fifty."
Assistant: One, two, three, four, five, six, seven, eight.
User: "Book an appointment for Rahul tomorrow at 10 AM, phone 9849012345."
Assistant: ACTION: {"tool": "book_appointment", "args": {"caller_name": "Rahul", "phone_number": "9849012345", "date": "tomorrow", "time_slot": "10 AM"}} Confirmed for Rahul tomorrow at 10 AM."""

def extract_action(text: str):
    """Extract complete JSON string from ACTION: {...} balancing nested braces."""
    if "ACTION:" not in text:
        return None, None
    idx = text.find("ACTION:")
    sub = text[idx + len("ACTION:"):].lstrip()
    if not sub.startswith("{"):
        return None, None
    depth = 0
    end_idx = -1
    for i, ch in enumerate(sub):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx != -1:
        json_str = sub[:end_idx]
        full_action_str = text[idx : idx + len("ACTION:") + (len(text[idx + len("ACTION:"):]) - len(sub)) + end_idx]
        return json_str, full_action_str
    return None, None

class SessionOrchestrator:
    def __init__(self, ws: WebSocket, stt: LocalSTT, tts: LocalTTS, vad: LocalVAD):
        self.ws = ws
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.llm = LocalLLM()
        self.is_active = True
        self.active_generation_task = None
        self.inbound_audio_buffer = bytearray()
        self.silence_chunks = 0
        self.is_speaking = False
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self):
        chunk_duration_ms = (settings.CHUNK_SIZE / settings.SAMPLE_RATE) * 1000
        silence_limit_chunks = int(settings.SILENCE_THRESHOLD_MS / chunk_duration_ms)

        try:
            while self.is_active:
                try:
                    msg = await self.ws.receive()
                except Exception:
                    break

                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    continue
                data = msg.get("bytes")
                if data is None:
                    continue

                has_speech = self.vad.is_speech(data)

                if has_speech:
                    if self.active_generation_task and not self.active_generation_task.done():
                        self.active_generation_task.cancel()
                        try:
                            await self.ws.send_json({"type": "interrupt"})
                        except Exception:
                            pass

                    self.is_speaking = True
                    self.silence_chunks = 0
                    self.inbound_audio_buffer.extend(data)
                else:
                    if self.is_speaking:
                        self.silence_chunks += 1
                        self.inbound_audio_buffer.extend(data)

                        if self.silence_chunks >= silence_limit_chunks:
                            self.is_speaking = False
                            turn_audio = bytes(self.inbound_audio_buffer)
                            self.inbound_audio_buffer.clear()
                            self.silence_chunks = 0

                            self.active_generation_task = asyncio.create_task(
                                self._execute_turn(turn_audio)
                            )
        finally:
            await self.llm.close()

    async def _execute_turn(self, pcm_bytes: bytes):
        t0 = time.perf_counter()

        # 1. Transcribe (tiny.en CUDA FP16)
        transcript = await asyncio.to_thread(self.stt.transcribe, pcm_bytes)
        if not transcript or len(transcript.strip()) < 2:
            return

        t_stt = (time.perf_counter() - t0) * 1000
        print(f"\n[User]: {transcript} (STT: {t_stt:.1f}ms)", flush=True)
        self.messages.append({"role": "user", "content": transcript})
        try:
            await self.ws.send_json({"type": "transcript", "role": "user", "text": transcript})
        except Exception:
            pass

        token_buffer = ""
        full_reply = ""
        first_audio_sent = False
        action_executed = False

        try:
            async for token in self.llm.stream_chat(self.messages):
                token_buffer += token
                full_reply += token

                # Parse deterministic tool execution inline
                if not action_executed and "ACTION:" in full_reply:
                    json_str, full_action_str = extract_action(full_reply)
                    if json_str:
                        try:
                            action_data = json.loads(json_str)
                            if action_data.get("tool") == "book_appointment":
                                args = action_data.get("args", {})
                                confirm_msg = book_appointment(**args)
                                action_executed = True
                                if confirm_msg:
                                    pcm_out = await asyncio.to_thread(self.tts.synthesize_to_pcm, confirm_msg)
                                    if pcm_out:
                                        if not first_audio_sent:
                                            total_ttfa = (time.perf_counter() - t0) * 1000
                                            print(f">>> [OPTIMIZED GLASS-TO-GLASS]: {total_ttfa:.1f} ms <<<", flush=True)
                                            first_audio_sent = True
                                        await self.ws.send_bytes(pcm_out)
                        except Exception as e:
                            print(f"[Tool Execution Error]: {e}", flush=True)
                        if full_action_str:
                            token_buffer = token_buffer.replace(full_action_str, "").strip()

                # If action already executed and spoken, do not repeat
                if action_executed:
                    token_buffer = ""
                    continue

                # Remove any residual action marker from token buffer
                cleaned_buffer = re.sub(r'ACTION:\s*\{.*?\}', '', token_buffer, flags=re.DOTALL).strip()
                if "ACTION:" in cleaned_buffer:
                    continue

                # Punctuation Shovel: Fire clauses immediately upon comma/period/delimiter or 3 words
                if (any(p in cleaned_buffer for p in [",", ".", "!", "?"]) or len(cleaned_buffer.split()) >= 3):
                    spoken_chunk = cleaned_buffer.strip()
                    if spoken_chunk and not spoken_chunk.startswith("ACTION"):
                        pcm_out = await asyncio.to_thread(self.tts.synthesize_to_pcm, spoken_chunk)
                        if pcm_out:
                            if not first_audio_sent:
                                total_ttfa = (time.perf_counter() - t0) * 1000
                                print(f">>> [OPTIMIZED GLASS-TO-GLASS]: {total_ttfa:.1f} ms <<<", flush=True)
                                first_audio_sent = True
                            await self.ws.send_bytes(pcm_out)
                    token_buffer = ""

            # Flush trailing audio tokens if action was not executed
            if not action_executed:
                cleaned_buffer = re.sub(r'ACTION:\s*\{.*?\}', '', token_buffer, flags=re.DOTALL).strip()
                if cleaned_buffer and not cleaned_buffer.startswith("ACTION"):
                    pcm_out = await asyncio.to_thread(self.tts.synthesize_to_pcm, cleaned_buffer)
                    if pcm_out:
                        if not first_audio_sent:
                            total_ttfa = (time.perf_counter() - t0) * 1000
                            print(f">>> [OPTIMIZED GLASS-TO-GLASS]: {total_ttfa:.1f} ms <<<", flush=True)
                            first_audio_sent = True
                        await self.ws.send_bytes(pcm_out)

            print(f"[Agent]: {full_reply}", flush=True)
            self.messages.append({"role": "assistant", "content": full_reply})
            try:
                await self.ws.send_json({"type": "transcript", "role": "assistant", "text": full_reply})
            except Exception:
                pass

        except asyncio.CancelledError:
            print("[Turn Interrupted via Barge-In]", flush=True)
        except Exception as e:
            print(f"[Pipeline Exception]: {e}", flush=True)
