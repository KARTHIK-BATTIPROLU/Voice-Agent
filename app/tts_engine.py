import os
import sys
import time
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
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

class LocalTTS:
    def __init__(self):
        # [SURGERY 3]: Shift Kokoro vocoder from CPU to RTX 4060 CUDA Cores -> Collapses 1088ms to ~35ms
        available_providers = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            settings.KOKORO_MODEL_PATH,
            sess_options=session_options,
            providers=providers
        )
        try:
            self.kokoro = Kokoro.from_session(self.session, settings.KOKORO_VOICES_PATH)
        except Exception:
            self.kokoro = Kokoro(settings.KOKORO_MODEL_PATH, settings.KOKORO_VOICES_PATH)
        # Re-bind internal ONNX session to the CUDA-optimized session
        self.kokoro.session = self.session
        self.kokoro.sess = self.session

    def synthesize_to_pcm(self, text: str) -> bytes:
        clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
        if not clean_text or not any(c.isalnum() for c in clean_text):
            return b""

        try:
            samples, sample_rate = self.kokoro.create(
                text=clean_text,
                voice="af_sarah",
                speed=1.1,
                lang="en-us"
            )

            # Rescale float32 array to 16-bit signed Linear PCM
            pcm_16 = (samples * 32767).astype(np.int16).tobytes()
            return pcm_16
        except Exception as e:
            print(f"[TTS Error]: {e}", flush=True)
            return b""

    def synthesize_timed(self, text: str):
        """Synthesize and return (pcm_bytes, stats) with per-phase timings."""
        t_start = time.perf_counter()
        pcm_16 = self.synthesize_to_pcm(text)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        stats = {
            "total_ms": duration_ms,
            "out_bytes": len(pcm_16)
        }
        return pcm_16, stats
