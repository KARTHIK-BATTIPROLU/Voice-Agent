import asyncio
import time
import json
import sqlite3
import numpy as np
import websockets
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app.tts_engine import LocalTTS

SERVER_URI = "ws://127.0.0.1:8000/ws/voice"
CHUNK_SIZE = 512
SAMPLE_RATE = 16000

async def verify_directive():
    print("--- [DIRECTIVE VERIFICATION]: Synthesizing target phrase ---")
    phrase = "Book an appointment for Rahul tomorrow at 10 AM, phone 9849012345."
    tts = LocalTTS()
    test_pcm = tts.synthesize_to_pcm(phrase)
    print(f"Synthesized {len(test_pcm)} bytes PCM.")

    # Record current row count in enterprise_ledger.db
    conn = sqlite3.connect("enterprise_ledger.db")
    c = conn.cursor()
    c.execute("SELECT count(*) FROM appointments")
    initial_count = c.fetchone()[0]
    print(f"Initial appointment count: {initial_count}")

    print("Connecting to live WebSocket server...")
    t_turn_start = None
    first_audio_received = False
    g2g_time_ms = 0.0

    async with websockets.connect(SERVER_URI) as ws:
        print("Connected! Streaming audio chunks...")
        t_turn_start = time.perf_counter()
        
        from app.vad import LocalVAD
        vad = LocalVAD()
        chunk_len = CHUNK_SIZE * 2
        raw_chunks = [test_pcm[i:i + chunk_len] for i in range(0, len(test_pcm), chunk_len)]
        if raw_chunks and len(raw_chunks[-1]) < chunk_len:
            raw_chunks[-1] += b"\x00" * (chunk_len - len(raw_chunks[-1]))

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

        for chunk in chunks:
            await ws.send(chunk)
            await asyncio.sleep(0.01)

        # Send silence chunks to trigger silence threshold (180ms = ~6 chunks)
        silence = b"\x00" * chunk_len
        for _ in range(15):
            await ws.send(silence)
            await asyncio.sleep(0.02)

        print("Finished sending audio. Awaiting response...")
        t_turn_start = time.perf_counter()
        received_audio_bytes = 0

        # Receive until we get audio and transcripts
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=4.0)
                if isinstance(msg, bytes):
                    if not first_audio_received:
                        g2g_time_ms = (time.perf_counter() - t_turn_start) * 1000
                        print(f">>> [CLIENT MEASURED TURNAROUND]: {g2g_time_ms:.1f} ms <<<")
                        first_audio_received = True
                    received_audio_bytes += len(msg)
                elif isinstance(msg, str):
                    event = json.loads(msg)
                    print(f"Received JSON event: {event}")
                    if event.get("role") == "assistant":
                        break
            except asyncio.TimeoutError:
                break

    # Verify Database Mutation
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments ORDER BY id DESC LIMIT 1")
    latest_row = cursor.fetchone()
    cursor.execute("SELECT count(*) FROM appointments")
    new_count = cursor.fetchone()[0]
    conn.close()

    print("\n================ VERIFICATION SUMMARY ================")
    print(f"Total audio received: {received_audio_bytes} bytes")
    print(f"Database mutation confirmed: {new_count > initial_count}")
    print(f"Latest Appointment in DB: {latest_row}")
    assert new_count > initial_count, "Database mutation failed: row count did not increase!"
    assert latest_row[1].lower() == "rahul", f"Expected caller_name 'Rahul', got '{latest_row[1]}'"
    print(">>> VERIFICATION CHECKLIST 100% COMPLETE AND PASSING! <<<")

if __name__ == "__main__":
    asyncio.run(verify_directive())
