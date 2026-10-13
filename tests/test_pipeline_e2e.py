import asyncio
import os
import sys
import numpy as np
import websockets
import json
import time

sys.path.insert(0, os.path.abspath("."))

SERVER_URI = "ws://127.0.0.1:8000/ws/voice"
SAMPLE_RATE = 16000
CHUNK_SIZE = 512

async def run_e2e_test():
    print("Connecting to sovereign voice websocket...")
    async with websockets.connect(SERVER_URI) as ws:
        print("Connected to WebSocket successfully.")

        # Generate a test audio utterance using LocalTTS or a known speech pattern
        from app.tts_engine import LocalTTS
        tts = LocalTTS()
        print("Synthesizing test input phrase: 'What is two plus two?'")
        test_pcm = tts.synthesize_to_pcm("What is two plus two?")
        assert len(test_pcm) > 0, "Failed to synthesize test PCM"
        print(f"Generated {len(test_pcm)} bytes of test speech PCM.")

        # Chunk audio into CHUNK_SIZE * 2 bytes (int16 mono)
        chunk_bytes_len = CHUNK_SIZE * 2
        chunks = [test_pcm[i:i+chunk_bytes_len] for i in range(0, len(test_pcm), chunk_bytes_len)]
        
        # Pad last chunk if needed
        if len(chunks[-1]) < chunk_bytes_len:
            chunks[-1] = chunks[-1] + b"\x00" * (chunk_bytes_len - len(chunks[-1]))

        print(f"Sending {len(chunks)} speech chunks to simulate speech turn...")
        for chunk in chunks:
            await ws.send(chunk)
            await asyncio.sleep(0.01)

        # Send silence chunks (e.g. 15 chunks = ~480ms > 250ms silence threshold) to trigger turn
        print("Sending silence chunks to trigger turn threshold...")
        silence_chunk = b"\x00" * chunk_bytes_len
        for _ in range(15):
            await ws.send(silence_chunk)
            await asyncio.sleep(0.02)

        print("Awaiting agent voice response from WebSocket...")
        received_audio_bytes = 0
        timeout = 15.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                if isinstance(msg, bytes):
                    received_audio_bytes += len(msg)
                    print(f"Received audio packet: {len(msg)} bytes (Total: {received_audio_bytes})")
                elif isinstance(msg, str):
                    payload = json.loads(msg)
                    print(f"Received JSON event: {payload}")
            except asyncio.TimeoutError:
                if received_audio_bytes > 0:
                    print("Turn completed. Received all audio.")
                    break

        print(f"Total response audio received: {received_audio_bytes} bytes")
        assert received_audio_bytes > 0, "No audio response received from pipeline!"
        print(">>> END-TO-END VERIFICATION PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
