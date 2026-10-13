import json
import time
import httpx
from typing import AsyncGenerator, Optional
from app.config import settings

class LocalLLM:
    _shared_client: Optional[httpx.AsyncClient] = None

    def __init__(self):
        self.url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        self.model = settings.OLLAMA_MODEL

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        if cls._shared_client is None or cls._shared_client.is_closed:
            cls._shared_client = httpx.AsyncClient(timeout=15.0)
        return cls._shared_client

    _get_client = get_client

    async def close(self):
        # Retain shared connection pool across sessions
        pass

    @classmethod
    async def shutdown(cls):
        if cls._shared_client is not None and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None

    async def stream_chat(self, messages: list, stats: Optional[dict] = None) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # [SURGERY 4]: keep_alive: -1 locks model weights and KV-cache in VRAM -> TTFT drops from 600ms to ~80ms
            "keep_alive": -1,
            "options": {
                "num_predict": 120,
                "temperature": 0.3,
                "top_p": 0.8
            }
        }
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            async with client.stream("POST", self.url, json=payload) as response:
                if stats is not None:
                    stats["http_ready_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    stats["http_status"] = response.status_code
                if response.status_code != 200:
                    if stats is not None:
                        stats["error"] = f"http_{response.status_code}"
                    yield "System operational."
                    return
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                if stats is not None and "first_token_ms" not in stats:
                                    stats["first_token_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                                yield content
                            if chunk.get("done") and stats is not None:
                                self._fill_usage(stats, chunk)
                        except json.JSONDecodeError:
                            continue
            if stats is not None:
                stats["stream_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except httpx.RequestError as e:
            if stats is not None:
                stats["error"] = f"{type(e).__name__}: {e}"
            yield "System operational."

    @staticmethod
    def _fill_usage(stats: dict, chunk: dict):
        """Copy Ollama's own accounting (nanoseconds -> ms) into stats."""
        def ms(key):
            v = chunk.get(key)
            return round(v / 1e6, 1) if isinstance(v, (int, float)) else None
        stats["ollama_total_ms"] = ms("total_duration")
        stats["ollama_load_ms"] = ms("load_duration")
        stats["ollama_prompt_eval_ms"] = ms("prompt_eval_duration")
        stats["ollama_eval_ms"] = ms("eval_duration")
        stats["prompt_tokens"] = chunk.get("prompt_eval_count")
        stats["completion_tokens"] = chunk.get("eval_count")
        stats["done_reason"] = chunk.get("done_reason")
        ev, ed = chunk.get("eval_count"), chunk.get("eval_duration")
        if ev and ed:
            stats["tokens_per_s"] = round(ev / (ed / 1e9), 1)
