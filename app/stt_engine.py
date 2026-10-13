import os
import sys
import time
import numpy as np
from faster_whisper import WhisperModel
from app.config import settings

# On Windows, ensure CUDA runtime DLLs (cublas, cudnn) installed via pip are loaded
if sys.platform == "win32":
    import site
    for site_pkg in site.getsitepackages():
        nvidia_path = os.path.join(site_pkg, "nvidia")
        if os.path.exists(nvidia_path):
            for root, _, files in os.walk(nvidia_path):
                if any(f.lower().endswith(".dll") for f in files):
                    try:
                        os.add_dll_directory(root)
                        if root not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                    except Exception:
                        pass

class LocalSTT:
    def __init__(self):
        print(f"Loading faster-whisper ({settings.WHISPER_MODEL}) onto RTX 4060 ({settings.WHISPER_DEVICE})...")
        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )

    def transcribe(self, pcm_bytes: bytes) -> str:
        return self.transcribe_timed(pcm_bytes)[0]

    def transcribe_timed(self, pcm_bytes: bytes):
        """Transcribe and return (text, stats). Stats carry per-phase timings measured inside the worker thread."""
        t_start = time.perf_counter()
        stats = {
            "model": settings.WHISPER_MODEL, "device": settings.WHISPER_DEVICE,
            "compute_type": settings.WHISPER_COMPUTE_TYPE, "in_bytes": len(pcm_bytes or b""),
        }
        if not pcm_bytes:
            stats.update(skipped="empty_audio", total_ms=0.0)
            return "", stats
        valid_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if valid_len < 1600:
            stats.update(skipped="audio_too_short", total_ms=0.0, audio_s=round(valid_len / 32000, 3))
            return "", stats
        audio_int16 = np.frombuffer(pcm_bytes[:valid_len], dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_s = len(audio_float32) / 16000.0
        stats["audio_s"] = round(audio_s, 3)
        stats["prep_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        try:
            t_model = time.perf_counter()
            segments, info = self.model.transcribe(
                audio_float32,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True
            )
            stats["setup_ms"] = round((time.perf_counter() - t_model) * 1000, 2)
            t_decode = time.perf_counter()
            seg_list = list(segments)  # faster-whisper is lazy: decoding happens while iterating
            stats["decode_ms"] = round((time.perf_counter() - t_decode) * 1000, 2)
            stats["segments"] = len(seg_list)
            stats["audio_after_vad_s"] = round(getattr(info, "duration_after_vad", audio_s) or 0.0, 3)
            stats["language_prob"] = round(getattr(info, "language_probability", 0.0) or 0.0, 3)
            text = " ".join(s.text for s in seg_list).strip()
        except Exception as e:
            print(f"[STT Error]: {e}", flush=True)
            stats["error"] = str(e)
            text = ""
        stats["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        stats["rtf"] = round((stats["total_ms"] / 1000.0) / audio_s, 3) if audio_s else 0.0
        return text, stats
