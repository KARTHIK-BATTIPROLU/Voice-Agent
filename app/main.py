import asyncio
import os
import platform
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from app import telemetry as tm
from app.config import settings

tm.setup()
tm.event("BOOT", "server_process_start", python=platform.python_version(), platform=platform.platform(),
         pid=os.getpid(), cwd=os.getcwd(), log_dir=tm.LOG_DIR)

tm.event("BOOT", "importing_models")
_t = time.perf_counter()
from app.stt_engine import LocalSTT
from app.tts_engine import LocalTTS
from app.vad import LocalVAD
from app.pipeline import SessionOrchestrator
tm.event("BOOT", "imports_done", ms=round((time.perf_counter() - _t) * 1000, 1))

print("Initializing sovereign neural models...")
_t = time.perf_counter()
shared_stt = LocalSTT()
tm.event("BOOT", "stt_loaded", ms=round((time.perf_counter() - _t) * 1000, 1),
         model=settings.WHISPER_MODEL, device=settings.WHISPER_DEVICE, compute=settings.WHISPER_COMPUTE_TYPE)
_t = time.perf_counter()
shared_tts = LocalTTS()
tm.event("BOOT", "tts_loaded", ms=round((time.perf_counter() - _t) * 1000, 1),
         model_path=settings.KOKORO_MODEL_PATH)
_t = time.perf_counter()
shared_vad = LocalVAD()
tm.event("BOOT", "vad_loaded", ms=round((time.perf_counter() - _t) * 1000, 1))

# One-time warm-up so the first real turn does not pay lazy-init costs; timings are logged separately.
try:
    import numpy as _np
    _t = time.perf_counter()
    shared_stt.transcribe_timed((_np.zeros(16000, dtype=_np.int16)).tobytes())
    tm.event("BOOT", "stt_warmup", ms=round((time.perf_counter() - _t) * 1000, 1))
    _t = time.perf_counter()
    _pcm, _st = shared_tts.synthesize_timed("Warm up.")
    tm.event("BOOT", "tts_warmup", ms=round((time.perf_counter() - _t) * 1000, 1), **{k: v for k, v in _st.items() if k != "clean_text"})
except Exception as _e:
    tm.event("BOOT", "warmup_failed", level="WARNING", error=str(_e))
tm.event("BOOT", "gpu_after_load", **tm.gpu_snapshot())

LOOP_LAG_INTERVAL_S = 0.05
LOOP_LAG_WARN_MS = 60.0


async def _loop_lag_monitor():
    """Detect event-loop stalls (anything blocking the loop delays mic chunks and barge-in)."""
    while True:
        t0 = time.perf_counter()
        await asyncio.sleep(LOOP_LAG_INTERVAL_S)
        lag_ms = (time.perf_counter() - t0 - LOOP_LAG_INTERVAL_S) * 1000
        if lag_ms > LOOP_LAG_WARN_MS:
            tm.event("LOOP", "event_loop_lag", level="WARNING", lag_ms=round(lag_ms, 1),
                     note="event loop was blocked; mic chunks and barge-in detection were delayed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_loop_lag_monitor())
    # Warm up Ollama model and preload SYSTEM_PROMPT KV-cache into VRAM
    try:
        from app.llm_client import LocalLLM
        from app.pipeline import SYSTEM_PROMPT
        warm_llm = LocalLLM()
        async for _ in warm_llm.stream_chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "ping"}]):
            pass
    except Exception as _e:
        print(f"[BOOT Warmup Error]: {_e}", flush=True)

    tm.event("BOOT", "server_ready", url="http://127.0.0.1:8000", ui="/", metrics="/metrics", logs="/logs")
    try:
        yield
    finally:
        task.cancel()
        from app.llm_client import LocalLLM
        await LocalLLM.shutdown()
        tm.event("BOOT", "server_shutdown")


app = FastAPI(title="Sovereign Edge Voice Server", lifespan=lifespan)


@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Sovereign Edge Voice Server is Running", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health_check():
    tts_provider = "kokoro-82m-cuda" if "CUDAExecutionProvider" in shared_tts.session.get_providers() else "kokoro-82m-cpu"
    return {
        "status": "operational",
        "stt": "faster-whisper-cuda",
        "llm": "ollama-qwen2.5-local",
        "tts": tts_provider,
        "cloud_apis": 0
    }


@app.get("/metrics")
async def metrics():
    """Aggregate stage timings (avg/p50/p95/min/max) over recent turns."""
    return JSONResponse(tm.aggregate())


@app.get("/logs")
async def logs(lines: int = 300):
    """Tail of the human-readable server log."""
    return PlainTextResponse(tm.tail_log(lines))


@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    t_conn = time.perf_counter()
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    ua = ws.headers.get("user-agent", "")
    tm.event("WS", "accepted", client=client, user_agent=ua[:100],
             accept_ms=round((time.perf_counter() - t_conn) * 1000, 1))
    from app.call_agent import JanbrightSessionOrchestrator
    session = JanbrightSessionOrchestrator(ws, shared_stt, shared_tts, shared_vad, audio_format="pcm16")
    try:
        await session.run()
    except WebSocketDisconnect:
        session.is_active = False
        tm.event("WS", "disconnect", level="INFO")
    except Exception as e:
        session.is_active = False
        tm.event("WS", "session_exception", level="WARNING", error=str(e))
        print(f"[Voice WS Exception]: {e}", flush=True)



@app.websocket("/ws/call-stream")
async def call_stream_websocket(ws: WebSocket, format: str = "pcm16"):
    """
    Audio Gateway Endpoint for Janbright Admissions Agent.
    Accepts 16kHz PCM chunks or 8kHz G.711 mu-law (via ?format=mulaw or JSON init).
    Executes Whisper STT -> LLM -> Kokoro TTS back down the socket with sub-400ms latency.
    """
    t_conn = time.perf_counter()
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    tm.event("WS", "janbright_call_stream_accepted", client=client, format=format,
             accept_ms=round((time.perf_counter() - t_conn) * 1000, 1))
    
    from app.call_agent import JanbrightSessionOrchestrator
    session = JanbrightSessionOrchestrator(ws, shared_stt, shared_tts, shared_vad, audio_format=format)
    try:
        await session.run()
    except WebSocketDisconnect:
        session.is_active = False
    except Exception as e:
        session.is_active = False
        print(f"[Janbright WS Exception]: {e}", flush=True)


# =====================================================================
# Janbright REST API Endpoints for Dashboard & Test Bench
# =====================================================================
from pydantic import BaseModel
from typing import Optional
from app.database import get_batches, get_leads, check_and_book_slot, reset_db
from app.knowledge_base import get_system_context, update_custom_rules

class BookingRequest(BaseModel):
    parent_name: str
    phone_number: str
    child_name: str
    grade: str
    activity: str
    slot_time: str

class SimulateCallRequest(BaseModel):
    phone_number: str = "9849012345"
    scenario: str = "fee_deflection_and_robotics"
    parent_name: str = "Srinivas Rao"
    child_name: str = "Aarav"
    grade: str = "Grade 6"

class ContextUpdateRequest(BaseModel):
    rules: str

class SynthesizeRequest(BaseModel):
    text: str

@app.get("/api/batches")
async def api_get_batches():
    return {"status": "ok", "batches": get_batches()}

@app.get("/api/leads")
async def api_get_leads(limit: int = 100):
    return {"status": "ok", "leads": get_leads(limit=limit)}

@app.post("/api/book")
async def api_book_slot(req: BookingRequest):
    result = check_and_book_slot(
        parent_name=req.parent_name,
        phone=req.phone_number,
        child=req.child_name,
        grade=req.grade,
        activity=req.activity,
        slot=req.slot_time
    )
    return result

@app.post("/api/simulate-call")
async def api_simulate_call(req: SimulateCallRequest):
    from app.call_agent import simulate_outbound_dial
    result = await simulate_outbound_dial(
        phone_number=req.phone_number,
        scenario=req.scenario,
        parent_name=req.parent_name,
        child_name=req.child_name,
        grade=req.grade
    )
    return result

@app.get("/api/context")
async def api_get_context():
    return {"status": "ok", "context": get_system_context()}

@app.post("/api/context")
async def api_update_context(req: ContextUpdateRequest):
    update_custom_rules(req.rules)
    return {"status": "ok", "context": get_system_context()}

@app.post("/api/reset-db")
async def api_reset_db():
    reset_db()
    return {"status": "ok", "batches": get_batches(), "leads": get_leads()}

@app.post("/api/synthesize")
async def api_synthesize(req: SynthesizeRequest):
    import base64
    pcm_bytes, stats = shared_tts.synthesize_timed(req.text)
    b64 = base64.b64encode(pcm_bytes).decode("ascii")
    return {"status": "ok", "stats": stats, "pcm_b64": b64, "sample_rate": 24000}
