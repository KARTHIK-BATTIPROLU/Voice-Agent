import asyncio
import time
import json
import re
import audioop
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from fastapi import WebSocket
from app.config import settings
from app.vad import LocalVAD
from app.stt_engine import LocalSTT
from app.llm_client import LocalLLM
from app.tts_engine import LocalTTS
from app.knowledge_base import get_system_context
from app.database import check_and_book_slot

def extract_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract complete JSON string from ACTION: {...} balancing nested braces.
    Supports case-insensitive variants (ACTION, Action, action) and markdown code fences.
    """
    if not text:
        return None, None
    match = re.search(r'action\s*:\s*(?:```(?:json)?\s*)?(\{)', text, re.IGNORECASE)
    if not match:
        return None, None
    start_brace_idx = match.start(1)
    full_start_idx = match.start()
    sub = text[start_brace_idx:]
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
        rest = text[start_brace_idx + end_idx:]
        trailing_ticks = re.match(r'^\s*```', rest)
        full_end_idx = start_brace_idx + end_idx + (trailing_ticks.end() if trailing_ticks else 0)
        full_action_str = text[full_start_idx:full_end_idx]
        return json_str, full_action_str
    return None, None

def get_agent_system_prompt() -> str:
    """Builds the dynamic system prompt with current date, enterprise context, and few-shots."""
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    school_context = get_system_context()
    prompt = (
        f"Today is {today_str}.\n"
        f"{school_context}\n\n"
        "DIALOGUE SCRIPT & BEHAVIOR GUIDE:\n"
        "- Turn 1: Warm greeting from Janbright World School, Mokila. Confirm parent name and ask about CAIE curriculum or complimentary Saturday workshops.\n"
        "- Turn 2: Help parent select between Robotics (Ages 7-14), Archery (Ages 9-16), or Swimming (Ages 6-15), and confirm the child's grade.\n"
        "- Turn 3: Time slot selection (Saturday 10:30 AM vs Saturday 03:00 PM). If parent asks about tuition fees, STRICTLY DEFLECT: state academic fees are discussed in detail during the in-person on-campus tour, but Saturday workshops are 100% complimentary.\n"
        "- Turn 4: Trigger tool execution.\n\n"
        "FEW-SHOT EXAMPLES:\n"
        "User: Hello, who is this?\n"
        "Assistant: Welcome to Janbright World School, Mokila! Are you inquiring about Cambridge admissions or our complimentary Saturday workshops for your child?\n\n"
        "User: How much does Grade 6 tuition cost?\n"
        "Assistant: Academic fee details are personalized and shared during your on-campus admissions meeting at Mokila. Our Saturday trial workshops are completely free—which workshop interests your child?\n\n"
        "User: Can I book Archery for Rohan, Grade 8 at 10:30 AM? Phone 9988776655.\n"
        'Assistant: ACTION: {"tool": "book_trial", "args": {"parent_name": "Parent", "phone": "9988776655", "child_name": "Rohan", "grade": "Grade 8", "activity": "Archery", "slot": "Saturday 10:30 AM"}} I am checking availability for Rohan right away.\n\n'
        "User: Book Robotics for Aarav, Grade 6, phone 9849012345 for Saturday 10:30 AM.\n"
        'Assistant: ACTION: {"tool": "book_trial", "args": {"parent_name": "Parent", "phone": "9849012345", "child_name": "Aarav", "grade": "Grade 6", "activity": "Robotics", "slot": "Saturday 10:30 AM"}} Confirmed! I have reserved Aarav\'s complimentary Robotics workshop for this Saturday at 10:30 AM.\n\n'
        "User: Count from 1 to 5.\n"
        "Assistant: 1, 2, 3, 4, 5.\n\n"
        "REMEMBER: Keep all spoken replies strictly under 2 sentences. No markdown formatting."
    )
    return prompt

class JanbrightCallAgent:
    """Conversational agent for admissions and trial booking."""
    def __init__(self):
        self.llm = LocalLLM()

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        stats: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Runs LLM stream, extracts ACTION: {...}, executes tool if present,
        and returns (spoken_text, action_dict, action_result).
        """
        full_reply = ""
        action_executed = False
        action_data = None
        action_result = None
        matched_action_str = None

        async for token in self.llm.stream_chat(messages, stats=stats):
            full_reply += token

            if not action_executed and re.search(r'action\s*:\s*', full_reply, re.IGNORECASE):
                json_str, full_action_str = extract_action(full_reply)
                if json_str:
                    matched_action_str = full_action_str
                    try:
                        action_data = json.loads(json_str)
                        tool_name = action_data.get("tool")
                        args = action_data.get("args", {})

                        if tool_name in ["book_trial", "book_workshop", "book_slot", "book_appointment"]:
                            # Execute deterministic ledger reservation
                            parent_name = args.get("parent_name") or args.get("caller_name") or "Parent"
                            phone = args.get("phone") or args.get("phone_number") or ""
                            child = args.get("child_name") or args.get("child") or "Student"
                            grade = args.get("grade") or "Grade 5"
                            activity = args.get("activity") or "Robotics"
                            slot = args.get("slot") or args.get("slot_time") or args.get("time_slot") or "Saturday 10:30 AM"

                            action_result = check_and_book_slot(
                                parent_name=parent_name,
                                phone=phone,
                                child=child,
                                grade=grade,
                                activity=activity,
                                slot=slot
                            )
                            action_executed = True
                    except Exception as e:
                        print(f"[Tool Execution Error]: {e}", flush=True)

        # Build clean spoken reply
        spoken_text = full_reply
        if action_result and action_result.get("message"):
            # When tool is executed, use the deterministic ledger response message
            spoken_text = action_result["message"]
        else:
            # Strip out extracted action or residual ACTION markup cleanly without leaving dangling braces
            if matched_action_str and matched_action_str in spoken_text:
                spoken_text = spoken_text.replace(matched_action_str, "").strip()
            spoken_text = re.sub(r'action\s*:\s*(?:```(?:json)?\s*)?\{.*?\}\s*(?:```)?', '', spoken_text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Enforce under 2 sentences constraint
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', spoken_text) if s.strip()]
        if len(sentences) > 2:
            spoken_text = " ".join(sentences[:2])

        return spoken_text, action_data, action_result

class JanbrightSessionOrchestrator:
    """
    WebSocket Audio Gateway Orchestrator for /ws/call-stream.
    Supports 16kHz PCM and 8kHz G.711 mu-law inbound audio,
    executing Whisper STT -> Ollama Qwen2.5-1.5B -> Kokoro TTS back down the socket.
    """
    def __init__(self, ws: WebSocket, stt: LocalSTT, tts: LocalTTS, vad: LocalVAD, audio_format: str = "pcm16"):
        self.ws = ws
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.agent = JanbrightCallAgent()
        self.is_active = True
        self.active_generation_task = None
        self.inbound_audio_buffer = bytearray()
        self.silence_chunks = 0
        self.is_speaking = False
        self.audio_format = audio_format  # "pcm16" or "mulaw"
        self.ratecv_state = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": get_agent_system_prompt()}
        ]

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

                # Support control JSON messages (format negotiation, prompt update, etc.)
                if msg.get("text") is not None:
                    try:
                        control = json.loads(msg["text"])
                        if control.get("type") == "init":
                            if control.get("format"):
                                self.audio_format = control["format"]
                        elif control.get("type") == "text_query":
                            # Allow direct text injection for test bench over socket
                            user_text = control.get("text", "")
                            if user_text:
                                self.active_generation_task = asyncio.create_task(
                                    self._execute_text_turn(user_text)
                                )
                    except Exception:
                        pass
                    continue

                raw_bytes = msg.get("bytes")
                if not raw_bytes:
                    continue

                # Inbound format adaptation: convert 8kHz G.711 mu-law to 16kHz PCM16 if needed
                if self.audio_format == "mulaw":
                    try:
                        pcm_8k = audioop.ulaw2lin(raw_bytes, 2)
                        pcm_16k, self.ratecv_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, self.ratecv_state)
                        data = pcm_16k
                    except Exception:
                        data = raw_bytes
                else:
                    data = raw_bytes

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
                                self._execute_audio_turn(turn_audio)
                            )
        finally:
            self.is_active = False

    async def _execute_text_turn(self, user_text: str):
        """Processes a text query over the WebSocket and emits speech and metrics."""
        t0 = time.perf_counter()
        self.messages.append({"role": "user", "content": user_text})
        try:
            await self.ws.send_json({"type": "transcript", "role": "user", "text": user_text})
        except Exception:
            pass

        llm_stats = {}
        t_llm_start = time.perf_counter()
        spoken_reply, action_data, action_result = await self.agent.generate_response(self.messages, stats=llm_stats)
        t_llm = (time.perf_counter() - t_llm_start) * 1000
        ttft = llm_stats.get("first_token_ms", t_llm)

        if action_data:
            try:
                await self.ws.send_json({"type": "action", "action": action_data, "result": action_result})
            except Exception:
                pass

        t_tts_start = time.perf_counter()
        pcm_out = await asyncio.to_thread(self.tts.synthesize_to_pcm, spoken_reply)
        t_tts = (time.perf_counter() - t_tts_start) * 1000
        total_g2g = (time.perf_counter() - t0) * 1000

        # Send metrics
        try:
            await self.ws.send_json({
                "type": "metrics",
                "stt_ms": 0.0,
                "llm_ttft_ms": round(ttft, 1),
                "tts_ms": round(t_tts, 1),
                "total_g2g_ms": round(total_g2g, 1)
            })
        except Exception:
            pass

        # Send audio down WebSocket
        if pcm_out:
            await self._send_audio_downstream(pcm_out)

        self.messages.append({"role": "assistant", "content": spoken_reply})
        try:
            await self.ws.send_json({"type": "transcript", "role": "assistant", "text": spoken_reply})
        except Exception:
            pass

    async def _execute_audio_turn(self, pcm_bytes: bytes):
        """Processes an inbound speech turn (Whisper STT -> Ollama LLM -> Kokoro TTS)."""
        t0 = time.perf_counter()

        # 1. Transcribe (tiny.en CUDA FP16)
        transcript, stt_stats = await asyncio.to_thread(self.stt.transcribe_timed, pcm_bytes)
        t_stt = stt_stats.get("total_ms", (time.perf_counter() - t0) * 1000)

        if not transcript or len(transcript.strip()) < 2:
            return

        print(f"\n[Janbright Caller]: {transcript} (STT: {t_stt:.1f}ms)", flush=True)
        self.messages.append({"role": "user", "content": transcript})
        try:
            await self.ws.send_json({"type": "transcript", "role": "user", "text": transcript})
        except Exception:
            pass

        # 2. LLM Turn
        llm_stats = {}
        t_llm_start = time.perf_counter()
        spoken_reply, action_data, action_result = await self.agent.generate_response(self.messages, stats=llm_stats)
        t_llm = (time.perf_counter() - t_llm_start) * 1000
        ttft = llm_stats.get("first_token_ms", t_llm)

        if action_data:
            try:
                await self.ws.send_json({"type": "action", "action": action_data, "result": action_result})
            except Exception:
                pass

        # 3. TTS Synthesis (Kokoro-82M CUDA)
        t_tts_start = time.perf_counter()
        pcm_out = await asyncio.to_thread(self.tts.synthesize_to_pcm, spoken_reply)
        t_tts = (time.perf_counter() - t_tts_start) * 1000
        total_g2g = (time.perf_counter() - t0) * 1000

        print(f"[Janbright Agent]: {spoken_reply} | TTFT: {ttft:.1f}ms | TTS: {t_tts:.1f}ms | Total G2G: {total_g2g:.1f}ms", flush=True)

        # 4. Stream Metrics
        try:
            await self.ws.send_json({
                "type": "metrics",
                "stt_ms": round(t_stt, 1),
                "llm_ttft_ms": round(ttft, 1),
                "tts_ms": round(t_tts, 1),
                "total_g2g_ms": round(total_g2g, 1)
            })
        except Exception:
            pass

        # 5. Send Audio
        if pcm_out:
            await self._send_audio_downstream(pcm_out)

        self.messages.append({"role": "assistant", "content": spoken_reply})
        try:
            await self.ws.send_json({"type": "transcript", "role": "assistant", "text": spoken_reply})
        except Exception:
            pass

    async def _send_audio_downstream(self, pcm_bytes: bytes):
        """Converts and sends audio packets matching client negotiated format."""
        try:
            if self.audio_format == "mulaw":
                # Resample 24000Hz (Kokoro default) or 16000Hz to 8000Hz and encode as mu-law
                pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, None)
                mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
                await self.ws.send_bytes(mulaw_bytes)
            else:
                await self.ws.send_bytes(pcm_bytes)
        except Exception as e:
            print(f"[WebSocket Audio Send Error]: {e}", flush=True)

async def simulate_outbound_dial(
    phone_number: str,
    scenario: str = "fee_deflection_and_robotics",
    parent_name: str = "Srinivas Rao",
    child_name: str = "Aarav",
    grade: str = "Grade 6"
) -> Dict[str, Any]:
    """
    Simulates an automated outbound admissions call flow for the Test Bench.
    Returns step-by-step transcript, latency telemetry, and database ledger mutation.
    """
    agent = JanbrightCallAgent()
    transcript_log = []
    messages = [{"role": "system", "content": get_agent_system_prompt()}]

    # Define scenario conversational turns
    if scenario == "fee_deflection_and_robotics":
        turns = [
            f"Hello, this is {parent_name}. I received an SMS about Janbright School Mokila.",
            "Can you tell me how much the Cambridge curriculum tuition fees are for Grade 6?",
            f"Alright, I'd like to book the complimentary Robotics workshop for my son {child_name}, {grade}, for Saturday 10:30 AM. My phone is {phone_number}."
        ]
    elif scenario == "archery_capacity_conflict":
        turns = [
            f"Hi, this is {parent_name} from Mokila.",
            f"I want to book an Archery trial workshop for {child_name}, {grade} this Saturday at 10:30 AM. Phone is {phone_number}."
        ]
    else:
        turns = [
            f"Hello, this is {parent_name}.",
            f"We are interested in the Cambridge curriculum and the Swimming workshop for {child_name}, {grade} this Saturday at 10:30 AM."
        ]

    total_pipeline_ms = 0.0
    action_result = None
    final_spoken = ""

    for user_turn in turns:
        t_turn_start = time.perf_counter()
        messages.append({"role": "user", "content": user_turn})
        stats = {}
        reply, action_data, res = await agent.generate_response(messages, stats=stats)
        turn_ms = (time.perf_counter() - t_turn_start) * 1000
        total_pipeline_ms += turn_ms

        transcript_log.append({
            "speaker": "Parent",
            "text": user_turn,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        transcript_log.append({
            "speaker": "Janbright Voice Agent",
            "text": reply,
            "latency_ms": round(turn_ms, 1),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        messages.append({"role": "assistant", "content": reply})
        if res:
            action_result = res
        final_spoken = reply

    return {
        "status": "COMPLETED",
        "phone_number": phone_number,
        "scenario": scenario,
        "transcript": transcript_log,
        "final_reply": final_spoken,
        "action_result": action_result,
        "total_turnaround_ms": round(total_pipeline_ms, 1)
    }
