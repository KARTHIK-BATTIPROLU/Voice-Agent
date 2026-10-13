import os
from pydantic import BaseModel

class AppConfig(BaseModel):
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 512  # 32ms frames
    
    # [SURGERY 1]: Prune Ear from base.en (74M) to tiny.en (39M) -> Saves ~230ms
    WHISPER_MODEL: str = "tiny.en"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    
    # [SURGERY 2]: Tighten VAD silence trigger from 250ms to 180ms -> Saves ~80ms
    SILENCE_THRESHOLD_MS: int = 180
    
    # Local Brain Parameters
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    
    # Paths for Local TTS Weights
    KOKORO_MODEL_PATH: str = os.path.abspath("models/kokoro-v0_19.onnx")
    KOKORO_VOICES_PATH: str = os.path.abspath("models/voices.bin")

settings = AppConfig()
