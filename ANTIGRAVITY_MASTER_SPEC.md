# TASK: 100% SELF-HOSTED, ZERO-API-KEY VOICE ORCHESTRATION PIPELINE
# RUNTIME TARGET: ASUS TUF A16 (NVIDIA RTX 4060 6GB VRAM, 16GB SYSTEM RAM, WSL2/UBUNTU)

## 1. HARD PHYSICAL CONSTRAINTS (ZERO DEVIATION PERMITTED)
1. ZERO External APIs: Do not import or call Deepgram, OpenAI, Cartesia, ElevenLabs, or Groq.
2. VRAM Ceiling (Hard Cap = 4.5 GB of 6 GB total VRAM):
   - Ear (STT): faster-whisper (model="base.en", device="cuda", compute_type="float16") -> ~1.2 GB VRAM.
   - Brain (LLM): Local Ollama instance running `qwen2.5:1.5b` (Q4_K_M quantization) -> ~1.5 GB VRAM.
   - Mouth (TTS): Kokoro-82M (via ONNX runtime on CPU to conserve GPU VRAM) -> 0 MB VRAM, ~350 MB RAM.
   - VAD: Silero VAD (ONNX runtime on CPU) -> 0 MB VRAM.
   - Total VRAM Consumption: ~2.7 GB to 3.0 GB. Leaves 3.0 GB of headroom for CUDA display context and buffers.
3. Audio Spec: Full-duplex Linear PCM 16kHz, 16-bit Mono (`pcm_s16le`). Buffer chunk = 512 samples (32ms slices).
4. Autonomous Verification: AntiGravity must write the codebase, pull weights, start the services, and verify execution using a local automated test client.

---

## 2. PROJECT DIRECTORY STRUCTURE
The workspace must match this exact layout:
sovereign-voice/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── vad.py
│   ├── stt_engine.py
│   ├── llm_client.py
│   ├── tts_engine.py
│   ├── pipeline.py
│   └── main.py
├── models/
├── scripts/
│   └── setup_env.sh
├── test_client.py
├── requirements.txt
└── run_server.sh

---

## 3. COMPONENT IMPLEMENTATION SPECIFICATIONS

### File: `requirements.txt`
```text
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
websockets>=12.0
faster-whisper>=1.0.0
kokoro-onnx>=0.2.0
onnxruntime>=1.17.0
sounddevice>=0.4.6
soundfile>=0.12.1
numpy>=1.26.0
httpx>=0.27.0
pydantic>=2.6.0
torch
torchaudio
```

### File: `scripts/setup_env.sh`

```bash
#!/usr/bin/env bash
set -e

echo "=== [1/4] Installing System Audio & Compilation Libraries ==="
sudo apt-get update && sudo apt-get install -y portaudio19-dev ffmpeg curl espeak-ng libsndfile1

echo "=== [2/4] Ensuring Ollama is Installed & Running ==="
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama service in the background if not active
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

echo "=== [3/4] Pulling Ultra-Fast 1.5B LLM (Zero VRAM Risk) ==="
ollama pull qwen2.5:1.5b

echo "=== [4/4] Downloading Kokoro-82M Weights ==="
mkdir -p models
if [ ! -f "models/kokoro-v0_19.onnx" ]; then
    curl -L -o models/kokoro-v0_19.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
fi
if [ ! -f "models/voices.bin" ]; then
    curl -L -o models/voices.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
fi

echo "=== Environment Setup Complete ==="
```

### File: `app/config.py`

```python
import os
from pydantic import BaseModel

class AppConfig(BaseModel):
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 512  # 32ms audio frame
    WHISPER_MODEL: str = "base.en"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    KOKORO_MODEL_PATH: str = os.path.abspath("models/kokoro-v0_19.onnx")
    KOKORO_VOICES_PATH: str = os.path.abspath("models/voices.bin")
    SILENCE_THRESHOLD_MS: int = 250  # VAD pause threshold to trigger response turn

settings = AppConfig()
```

### File: `app/vad.py`

```python
import torch
import numpy as np

class LocalVAD:
    """Zero-VRAM Voice Activity Detection using CPU Silero."""
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils

    def is_speech(self, pcm_chunk: bytes) -> bool:
        audio_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor_chunk = torch.from_numpy(audio_float32)
        speech_prob = self.model(tensor_chunk, 16000).item()
        return speech_prob > 0.5
```

### File: `app/stt_engine.py`

```python
import numpy as np
from faster_whisper import WhisperModel
from app.config import settings

class LocalSTT:
    def __init__(self):
        print(f"Loading faster-whisper ({settings.WHISPER_MODEL}) onto RTX 4060 ({settings.WHISPER_DEVICE})...")
        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )

    def transcribe(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes:
            return ""
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float32,
            beam_size=1,
            language="en",
            condition_on_previous_text=False
        )
        return " ".join([s.text for s in segments]).strip()
```

### File: `app/llm_client.py`

```python
import httpx
from typing import AsyncGenerator
from app.config import settings

class LocalLLM:
    def __init__(self):
        self.url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        self.model = settings.OLLAMA_MODEL

    async def stream_chat(self, messages: list) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": 25,
                "temperature": 0.4
            }
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream("POST", self.url, json=payload) as response:
                if response.status_code != 200:
                    yield "System error."
                    return
                async for line in response.aiter_lines():
                    if line:
                        import json
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
```

### File: `app/tts_engine.py`

```python
import numpy as np
from kokoro_onnx import Kokoro
from app.config import settings

class LocalTTS:
    def __init__(self):
        print("Initializing Kokoro-82M on host CPU...")
        self.kokoro = Kokoro(settings.KOKORO_MODEL_PATH, settings.KOKORO_VOICES_PATH)

    def synthesize_to_pcm(self, text: str) -> bytes:
        clean_text = text.strip()
        if not clean_text:
            return b""
        samples, sample_rate = self.kokoro.create(
            text=clean_text,
            voice="af_sarah",
            speed=1.1,
            lang="en-us"
        )
        pcm_16 = (samples * 32767).astype(np.int16).tobytes()
        return pcm_16
```

### File: `app/pipeline.py`

```python
import asyncio
import time
from fastapi import WebSocket
from app.config import settings
from app.vad import LocalVAD
from app.stt_engine import LocalSTT
from app.llm_client import LocalLLM
from app.tts_engine import LocalTTS

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
        self.messages = [
            {
                "role": "system",
                "content": "You are a sharp, concise voice operator. Keep responses under 8 words. No markdown."
            }
        ]

    async def run(self):
        chunk_duration_ms = (settings.CHUNK_SIZE / settings.SAMPLE_RATE) * 1000
        silence_limit_chunks = int(settings.SILENCE_THRESHOLD_MS / chunk_duration_ms)

        while self.is_active:
            try:
                data = await self.ws.receive_bytes()
            except Exception:
                break

            # 1. Local VAD Evaluation
            has_speech = self.vad.is_speech(data)

            if has_speech:
                # Interruption Handling (Barge-in)
                if self.active_generation_task and not self.active_generation_task.done():
                    self.active_generation_task.cancel()
                    await self.ws.send_json({"type": "interrupt"})

                self.is_speaking = True
                self.silence_chunks = 0
                self.inbound_audio_buffer.extend(data)
            else:
                if self.is_speaking:
                    self.silence_chunks += 1
                    self.inbound_audio_buffer.extend(data)

                    # Silence limit reached -> Trigger turn
                    if self.silence_chunks >= silence_limit_chunks:
                        self.is_speaking = False
                        turn_audio = bytes(self.inbound_audio_buffer)
                        self.inbound_audio_buffer.clear()
                        self.silence_chunks = 0

                        self.active_generation_task = asyncio.create_task(
                            self._execute_turn(turn_audio)
                        )

    async def _execute_turn(self, pcm_bytes: bytes):
        t0 = time.perf_counter()
        # 1. Transcribe
        transcript = self.stt.transcribe(pcm_bytes)
        if not transcript or len(transcript.strip()) < 2:
            return

        print(f"\n[User]: {transcript}")
        self.messages.append({"role": "user", "content": transcript})

        # 2. Stream LLM with Token-Shovel
        token_buffer = ""
        full_reply = ""

        try:
            async for token in self.llm.stream_chat(self.messages):
                token_buffer += token
                full_reply += token

                # Fire to TTS upon punctuation delimiter or 3 words
                if any(p in token_buffer for p in [",", ".", "!", "?"]) or len(token_buffer.split()) >= 3:
                    pcm_out = self.tts.synthesize_to_pcm(token_buffer)
                    if pcm_out:
                        await self.ws.send_bytes(pcm_out)
                    token_buffer = ""

            if token_buffer.strip():
                pcm_out = self.tts.synthesize_to_pcm(token_buffer)
                if pcm_out:
                    await self.ws.send_bytes(pcm_out)

            print(f"[Agent ({((time.perf_counter() - t0)*1000):.1f}ms)]: {full_reply}")
            self.messages.append({"role": "assistant", "content": full_reply})

        except asyncio.CancelledError:
            print("[Turn Aborted via Barge-In]")
        except Exception as e:
            print(f"[Pipeline Error]: {e}")
```

### File: `app/main.py`

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.stt_engine import LocalSTT
from app.tts_engine import LocalTTS
from app.vad import LocalVAD
from app.pipeline import SessionOrchestrator

app = FastAPI(title="Sovereign Edge Voice Server")

print("Initializing sovereign neural models...")
shared_stt = LocalSTT()
shared_tts = LocalTTS()
shared_vad = LocalVAD()

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "stt": "faster-whisper-cuda",
        "llm": "ollama-qwen2.5-local",
        "tts": "kokoro-82m-cpu",
        "cloud_apis": 0
    }

@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    await ws.accept()
    session = SessionOrchestrator(ws, shared_stt, shared_tts, shared_vad)
    try:
        await session.run()
    except WebSocketDisconnect:
        session.is_active = False
```

### File: `test_client.py`

```python
import asyncio
import sys
import websockets
import sounddevice as sd
import numpy as np
import json

SERVER_URI = "ws://127.0.0.1:8000/ws/voice"
SAMPLE_RATE = 16000
CHUNK_SIZE = 512

async def audio_harness():
    async with websockets.connect(SERVER_URI) as ws:
        print("\n>>> Sovereign Pipeline Connected. Speak into your microphone. Press Ctrl+C to exit. <<<\n")

        loop = asyncio.get_running_loop()
        in_queue = asyncio.Queue()

        def mic_callback(indata, frames, time, status):
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
                        speaker_stream.write(audio_data)
                    elif isinstance(msg, str):
                        payload = json.loads(msg)
                        if payload.get("type") == "interrupt":
                            print("\n[Barge-in Triggered]: Cleared audio queue.")

            await asyncio.gather(send_mic(), play_speaker())

if __name__ == "__main__":
    try:
        asyncio.run(audio_harness())
    except KeyboardInterrupt:
        print("\nSession Terminated.")
        sys.exit(0)
```

### File: `run_server.sh`

```bash
#!/usr/bin/env bash
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

---

## 4. EXECUTION DIRECTIVE FOR ANTIGRAVITY

AntiGravity must iteratively perform these steps in order:

1. Run `bash scripts/setup_env.sh` to download models and verify Ollama.
2. Create Python virtual environment (`python3 -m venv venv`), activate it, and install `requirements.txt`.
3. Generate all application files under `app/`.
4. Launch `uvicorn app.main:app --host 127.0.0.1 --port 8000` in the background.
5. Query `http://127.0.0.1:8000/health` to confirm `status == operational`.
6. Terminate only when step 5 returns HTTP 200 with all components initialized.
