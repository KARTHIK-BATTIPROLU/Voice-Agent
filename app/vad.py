import torch
import numpy as np

class LocalVAD:
    """Zero-VRAM Voice Activity Detection using CPU Silero."""
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils

    def reset_states(self):
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def is_speech(self, pcm_chunk: bytes) -> bool:
        return self.speech_prob(pcm_chunk) > 0.5

    def speech_prob(self, pcm_chunk: bytes) -> float:
        """Return the maximum Silero speech probability over the 512-sample windows in this chunk."""
        if not pcm_chunk or len(pcm_chunk) < 2:
            return 0.0
        valid_len = len(pcm_chunk) - (len(pcm_chunk) % 2)
        audio_int16 = np.frombuffer(pcm_chunk[:valid_len], dtype=np.int16)
        if len(audio_int16) == 0:
            return 0.0

        CHUNK = 512
        max_prob = 0.0
        try:
            for i in range(0, len(audio_int16), CHUNK):
                slice_arr = audio_int16[i:i + CHUNK]
                if len(slice_arr) < CHUNK:
                    slice_arr = np.pad(slice_arr, (0, CHUNK - len(slice_arr)))
                audio_float32 = slice_arr.astype(np.float32) / 32768.0
                tensor_chunk = torch.from_numpy(audio_float32)
                speech_prob = self.model(tensor_chunk, 16000).item()
                if speech_prob > max_prob:
                    max_prob = speech_prob
            return max_prob
        except Exception:
            return 0.0
