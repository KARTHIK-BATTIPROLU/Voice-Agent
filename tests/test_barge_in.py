import asyncio
import os
import sys
import websockets
import json

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
async def test_barge_in():
    async with websockets.connect(SERVER_URI) as ws:
        from app.tts_engine import LocalTTS
        tts = LocalTTS()

        # Ask a question that produces longer speech
        pcm_ask = tts.synthesize_to_pcm("Please count from one to twenty slowly.")
        chunk_len = CHUNK_SIZE * 2
        chunks = [pcm_ask[i:i+chunk_len] for i in range(0, len(pcm_ask), chunk_len)]
        if len(chunks[-1]) < chunk_len:
            chunks[-1] += b"\x00" * (chunk_len - len(chunks[-1]))

        # Send speech chunks
        for chunk in chunks:
            await ws.send(chunk)
            await asyncio.sleep(0.01)

        # Send silence to trigger turn
        for _ in range(12):
            await ws.send(b"\x00" * chunk_len)
            await asyncio.sleep(0.02)

        # Now simulate user interrupting (barge-in) by sending speech chunks again immediately
        pcm_interrupt = tts.synthesize_to_pcm("Stop!")
        int_chunks = [pcm_interrupt[i:i+chunk_len] for i in range(0, len(pcm_interrupt), chunk_len)]
        if len(int_chunks[-1]) < chunk_len:
            int_chunks[-1] += b"\x00" * (chunk_len - len(int_chunks[-1]))

        interrupt_received = False
        async def send_interruption():
            nonlocal interrupt_received
            await asyncio.sleep(0.5)
            print("Sending barge-in speech...")
            for c in int_chunks:
                await ws.send(c)
                await asyncio.sleep(0.01)

        async def listen_for_interrupt():
            nonlocal interrupt_received
            for _ in range(30):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(msg, str):
                        payload = json.loads(msg)
                        if payload.get("type") == "interrupt":
                            print("Received interrupt event from server!")
                            interrupt_received = True
                            break
                except asyncio.TimeoutError:
                    pass

        await asyncio.gather(send_interruption(), listen_for_interrupt())
        assert interrupt_received, "Failed to receive barge-in interrupt signal!"
        print(">>> BARGE-IN INTERRUPTION TEST PASSED! <<<")

if __name__ == "__main__":
    asyncio.run(test_barge_in())
