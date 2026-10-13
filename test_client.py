import asyncio
import sys
import websockets
import sounddevice as sd
import numpy as np
import json
import time

SERVER_URI = "ws://127.0.0.1:8000/ws/voice"
SAMPLE_RATE = 16000
CHUNK_SIZE = 512

async def audio_harness(phrase: str = None):
    async with websockets.connect(SERVER_URI) as ws:
        print("\n>>> Sovereign Pipeline Connected. <<<\n", flush=True)

        if phrase:
            from app.tts_engine import LocalTTS
            from app.vad import LocalVAD
            tts = LocalTTS()
            vad = LocalVAD()
            print(f"[Simulator]: Synthesizing speech for test phrase: '{phrase}'", flush=True)
            test_pcm = tts.synthesize_to_pcm(phrase)
            chunk_len = CHUNK_SIZE * 2
            raw_chunks = [test_pcm[i:i+chunk_len] for i in range(0, len(test_pcm), chunk_len)]
            if raw_chunks and len(raw_chunks[-1]) < chunk_len:
                raw_chunks[-1] += b"\x00" * (chunk_len - len(raw_chunks[-1]))

            # Cap intra-utterance silence gaps to 3 chunks (< 180ms threshold) so VAD doesn't split sentences
            chunks = []
            silence_streak = 0
            for c in raw_chunks:
                if vad.is_speech(c):
                    silence_streak = 0
                    chunks.append(c)
                else:
                    silence_streak += 1
                    if silence_streak <= 3:
                        chunks.append(c)

            print(f"[Simulator]: Sending {len(chunks)} audio chunks...", flush=True)
            for c in chunks:
                await ws.send(c)
                await asyncio.sleep(0.01)

            print("[Simulator]: Sending silence chunks to trigger turn...", flush=True)
            for _ in range(15):
                await ws.send(b"\x00" * chunk_len)
                await asyncio.sleep(0.02)

            # Listen for responses
            received_audio_bytes = 0
            t_end = time.time() + 15.0
            while time.time() < t_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    if isinstance(msg, bytes):
                        received_audio_bytes += len(msg)
                    elif isinstance(msg, str):
                        payload = json.loads(msg)
                        if payload.get("type") == "transcript":
                            role = payload.get("role", "agent").capitalize()
                            print(f"\n[{role}]: {payload.get('text', '')}", flush=True)
                        elif payload.get("type") == "turn_metrics":
                            stages = payload.get("stages_ms", {})
                            print(f"\n[Turn #{payload.get('turn')} Latency Breakdown]:", flush=True)
                            for k, v in stages.items():
                                print(f"  - {k:<26}: {v:>6.1f} ms", flush=True)
                except asyncio.TimeoutError:
                    if received_audio_bytes > 0:
                        break
            print(f"\nTurn complete. Received {received_audio_bytes} bytes of synthesized audio.", flush=True)
            return

        print("Speak into your microphone. Press Ctrl+C to exit.\n", flush=True)
        loop = asyncio.get_running_loop()
        in_queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(in_queue.put_nowait, indata.copy())

        mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_SIZE,
            callback=mic_callback
        )
        speaker_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_SIZE
        )

        with mic_stream, speaker_stream:
            async def send_mic():
                while True:
                    data = await in_queue.get()
                    await ws.send(data.tobytes())

            async def play_speaker():
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        audio_data = np.frombuffer(msg, dtype=np.int16)
                        try:
                            await ws.send(json.dumps({"type": "client_event", "event": "playback_started"}))
                        except Exception:
                            pass
                        await asyncio.to_thread(speaker_stream.write, audio_data)
                    elif isinstance(msg, str):
                        payload = json.loads(msg)
                        if payload.get("type") == "interrupt":
                            print("\n[Barge-in Triggered]: Cleared audio queue.", flush=True)
                            try:
                                speaker_stream.stop()
                                speaker_stream.start()
                            except Exception:
                                pass
                        elif payload.get("type") == "transcript":
                            role = payload.get("role", "agent").capitalize()
                            text = payload.get("text", "")
                            print(f"\n[{role}]: {text}", flush=True)
                        elif payload.get("type") == "turn_metrics":
                            stages = payload.get("stages_ms", {})
                            print(f"\n[Turn #{payload.get('turn')} Latency Breakdown]:", flush=True)
                            for k, v in stages.items():
                                print(f"  - {k:<26}: {v:>6.1f} ms", flush=True)

            await asyncio.gather(send_mic(), play_speaker())

if __name__ == "__main__":
    phrase_arg = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--phrase" and len(sys.argv) > 2:
            phrase_arg = " ".join(sys.argv[2:])
        elif not sys.argv[1].startswith("--"):
            phrase_arg = " ".join(sys.argv[1:])
    try:
        asyncio.run(audio_harness(phrase_arg))
    except KeyboardInterrupt:
        print("\nSession Terminated.")
        sys.exit(0)
