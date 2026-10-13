import asyncio
import os
import sys
import json
import time
import websockets
import numpy as np

sys.path.insert(0, os.path.abspath("."))

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            @staticmethod
            def asyncio(fn): return fn
    pytest = _MockPytest()

SERVER_URI_PCM = "ws://127.0.0.1:8000/ws/call-stream?format=pcm16"
SERVER_URI_MULAW = "ws://127.0.0.1:8000/ws/call-stream?format=mulaw"

@pytest.mark.asyncio
async def test_call_stream_text_control():
    """Verify text injection over /ws/call-stream WebSocket."""
    print("\n--- [Test 1] Testing /ws/call-stream text injection & metrics ---")
    async with websockets.connect(SERVER_URI_PCM) as ws:
        # Send text control query
        query_payload = {
            "type": "text_query",
            "text": "What is the fee for Grade 6 Cambridge curriculum?"
        }
        await ws.send(json.dumps(query_payload))

        received_audio = bytearray()
        received_metrics = None
        assistant_reply = None

        t_end = time.time() + 10.0
        while time.time() < t_end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(msg, bytes):
                    received_audio.extend(msg)
                elif isinstance(msg, str):
                    evt = json.loads(msg)
                    if evt.get("type") == "metrics":
                        received_metrics = evt
                    elif evt.get("type") == "transcript" and evt.get("role") == "assistant":
                        assistant_reply = evt.get("text")
                        break
            except asyncio.TimeoutError:
                if len(received_audio) > 0 and assistant_reply:
                    break

        print(f"Assistant Reply: {assistant_reply}")
        print(f"Metrics Received: {received_metrics}")
        print(f"Audio Bytes Received: {len(received_audio)}")

        assert assistant_reply is not None, "Did not receive assistant reply"
        assert len(received_audio) > 0, "Did not receive synthesized audio"
        assert received_metrics is not None, "Did not receive metrics telemetry"
        assert "fee" in assistant_reply.lower() or "campus" in assistant_reply.lower()
        print("PASS: /ws/call-stream text control and TTS streaming verified!")

@pytest.mark.asyncio
async def test_call_stream_mulaw_negotiation():
    """Verify G.711 mu-law audio negotiation."""
    print("\n--- [Test 2] Testing /ws/call-stream with G.711 mu-law format ---")
    async with websockets.connect(SERVER_URI_MULAW) as ws:
        query_payload = {
            "type": "text_query",
            "text": "Tell me about the Robotics workshop."
        }
        await ws.send(json.dumps(query_payload))

        received_audio = bytearray()
        assistant_reply = None

        t_end = time.time() + 10.0
        while time.time() < t_end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(msg, bytes):
                    received_audio.extend(msg)
                elif isinstance(msg, str):
                    evt = json.loads(msg)
                    if evt.get("type") == "transcript" and evt.get("role") == "assistant":
                        assistant_reply = evt.get("text")
                        break
            except asyncio.TimeoutError:
                if len(received_audio) > 0 and assistant_reply:
                    break

        print(f"Mu-law Assistant Reply: {assistant_reply}")
        print(f"Mu-law Audio Bytes Received: {len(received_audio)}")
        assert assistant_reply is not None
        assert len(received_audio) > 0
        print("PASS: G.711 mu-law telephony audio stream verified!")

async def main():
    await test_call_stream_text_control()
    await test_call_stream_mulaw_negotiation()
    print("\n>>> ALL WEBSOCKET AUDIO GATEWAY TESTS PASSED! <<<\n")

if __name__ == "__main__":
    asyncio.run(main())
