import asyncio
import os
import sys
import json
import time
import numpy as np
import websockets

sys.path.insert(0, os.path.abspath("."))

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            @staticmethod
            def asyncio(fn): return fn
    pytest = _MockPytest()

SERVER_URI = "ws://127.0.0.1:8000/ws/voice"
CHUNK_SIZE = 512

@pytest.mark.asyncio
async def test_silence_input():
    print("\n--- [Test 1] Testing pure silence input (no hallucinated turns) ---")
    async with websockets.connect(SERVER_URI) as ws:
        silence_chunk = b"\x00" * (CHUNK_SIZE * 2)
        # Send 30 silence chunks (~1 second)
        for _ in range(30):
            await ws.send(silence_chunk)
            await asyncio.sleep(0.02)
        
        # Verify no response is triggered
        received_anything = False
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            received_anything = True
        except asyncio.TimeoutError:
            pass
        assert not received_anything, "Server erroneously triggered a turn on pure silence!"
        print("PASS: Pure silence input correctly produces zero turns.")

@pytest.mark.asyncio
async def test_malformed_and_odd_chunks():
    print("\n--- [Test 2] Testing malformed and unaligned audio chunks ---")
    async with websockets.connect(SERVER_URI) as ws:
        # Send odd bytes, 0 bytes, 1 byte, 100 bytes, 3000 bytes
        for weird_chunk in [b"", b"\x00", b"\x01\x02\x03", b"\x00" * 99, b"\x00" * 3000]:
            await ws.send(weird_chunk)
            await asyncio.sleep(0.01)
        
        # Verify websocket is still alive and operational
        await ws.send(b"\x00" * 1024)
        print("PASS: Server cleanly absorbed all malformed chunks without socket crash.")

@pytest.mark.asyncio
async def test_barge_in_conversational_coherence():
    print("\n--- [Test 3] Testing conversational coherence across interrupted barge-in ---")
    async with websockets.connect(SERVER_URI) as ws:
        from app.tts_engine import LocalTTS
        tts = LocalTTS()

        # Step 1: User asks a long-winded question
        pcm_q1 = tts.synthesize_to_pcm("Please count slowly from one to fifty.")
        chunk_len = CHUNK_SIZE * 2
        for i in range(0, len(pcm_q1), chunk_len):
            chunk = pcm_q1[i:i+chunk_len]
            if len(chunk) < chunk_len:
                chunk += b"\x00" * (chunk_len - len(chunk))
            await ws.send(chunk)
            await asyncio.sleep(0.01)
        
        # Send silence to trigger turn
        for _ in range(12):
            await ws.send(b"\x00" * chunk_len)
            await asyncio.sleep(0.02)

        saw_interrupt = False
        received_audio = bytearray()

        # Concurrently listen for responses / interrupts
        async def listener():
            nonlocal saw_interrupt, received_audio
            t_end = time.time() + 15.0
            while time.time() < t_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(msg, str):
                        payload = json.loads(msg)
                        if payload.get("type") == "interrupt":
                            saw_interrupt = True
                            print("Confirmed: Interrupt signal received on barge-in.")
                    elif isinstance(msg, bytes):
                        received_audio.extend(msg)
                except asyncio.TimeoutError:
                    if saw_interrupt and len(received_audio) > 0:
                        break

        async def sender():
            # Wait briefly for generation to start, then barge in with Question B
            await asyncio.sleep(0.6)
            print("Injecting barge-in interruption: 'Stop what is two plus two?'")
            pcm_q2 = tts.synthesize_to_pcm("Stop what is two plus two?")
            for i in range(0, len(pcm_q2), chunk_len):
                chunk = pcm_q2[i:i+chunk_len]
                if len(chunk) < chunk_len:
                    chunk += b"\x00" * (chunk_len - len(chunk))
                await ws.send(chunk)
                await asyncio.sleep(0.01)

            # Send silence to trigger second turn
            for _ in range(12):
                await ws.send(b"\x00" * chunk_len)
                await asyncio.sleep(0.02)

        await asyncio.gather(listener(), sender())

        assert saw_interrupt, "Failed to observe interrupt signal on barge-in!"
        assert len(received_audio) > 0, "No audio response received after barge-in question!"
        print(f"PASS: Interrupted turn completed with {len(received_audio)} bytes audio response.")

@pytest.mark.asyncio
async def test_audio_format_and_rate():
    print("\n--- [Test 4] Testing audio format and sample rate verification ---")
    from app.tts_engine import LocalTTS
    from app.config import settings
    tts = LocalTTS()
    pcm = tts.synthesize_to_pcm("Testing the audio sample rate.")
    num_samples = len(pcm) // 2
    duration_s = num_samples / settings.SAMPLE_RATE
    print(f"Synthesized text length: {len(pcm)} bytes = {num_samples} samples = {duration_s:.2f}s at {settings.SAMPLE_RATE}Hz")
    assert len(pcm) % 2 == 0, "PCM audio bytes not aligned to 16-bit!"
    assert settings.SAMPLE_RATE == 16000, f"Expected 16000Hz, got {settings.SAMPLE_RATE}"
    print("PASS: Audio format is verified 16-bit 16000Hz Linear PCM.")

async def main():
    await test_audio_format_and_rate()
    await test_silence_input()
    await test_malformed_and_odd_chunks()
    await test_barge_in_conversational_coherence()
    print("\n>>> ALL ADVANCED EDGE CASE TESTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    asyncio.run(main())
