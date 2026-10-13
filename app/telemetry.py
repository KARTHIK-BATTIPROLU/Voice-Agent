"""
Telemetry for the Sovereign Voice pipeline.

Three outputs are written under <project>/logs/ :
  server.log   - human-readable, one line per event (also echoed to console)
  events.jsonl - every event as one JSON object (machine readable, full detail)
  turns.jsonl  - one JSON summary per conversational turn (stage timings, text, stats)

All console/file text is plain ASCII on purpose (Windows consoles choke on box characters).
"""
import datetime
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
EVENTS_JSONL = os.path.join(LOG_DIR, "events.jsonl")
TURNS_JSONL = os.path.join(LOG_DIR, "turns.jsonl")

_lock = threading.Lock()
_logger = None
_recent_turns = deque(maxlen=500)

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def setup():
    """Idempotently configure log files and the console handler."""
    global _logger
    if _logger is not None:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("sovereign")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s", "%H:%M:%S")
    file_handler = RotatingFileHandler(SERVER_LOG, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    _logger = logger


def _fmt_value(v):
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else str(v)
    if isinstance(v, str):
        s = v if len(v) <= 140 else v[:137] + "..."
        return json.dumps(s, ensure_ascii=True)
    return str(v)


def _append_jsonl(path, record):
    line = json.dumps(record, default=str, ensure_ascii=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def event(component, name, sid=None, turn=None, level="INFO", **fields):
    """Record one pipeline event to console, server.log and events.jsonl."""
    setup()
    record = {
        "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "epoch": round(time.time(), 3),
        "level": level,
        "sid": sid,
        "turn": turn,
        "component": component,
        "event": name,
    }
    record.update(fields)
    try:
        _append_jsonl(EVENTS_JSONL, record)
    except Exception:
        pass
    ctx = f"[{sid or '-'}{('/T' + str(turn)) if turn else ''}]"
    kv = " ".join(f"{k}={_fmt_value(v)}" for k, v in fields.items())
    _logger.log(_LEVELS.get(level, logging.INFO), f"{ctx:<9} {component:<9} {name:<22} {kv}")


def gpu_snapshot():
    """Query nvidia-smi (blocking, ~50-150ms). Call from a worker thread, never in the hot path."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=4,
        ).stdout.strip().splitlines()[0]
        used, total, util, temp, sm_clock, power = [x.strip() for x in out.split(",")]
        return {
            "vram_used_mb": float(used), "vram_total_mb": float(total), "gpu_util_pct": float(util),
            "gpu_temp_c": float(temp), "sm_clock_mhz": float(sm_clock), "power_w": power,
        }
    except Exception as e:
        return {"error": str(e)}


def record_turn(summary):
    _recent_turns.append(summary)
    try:
        _append_jsonl(TURNS_JSONL, summary)
    except Exception:
        pass


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo), 1)


def aggregate():
    """Aggregate stage timings across completed turns kept in memory."""
    done = [t for t in _recent_turns if t.get("outcome") == "completed"]
    keys = set()
    for t in done:
        keys.update(t.get("stages_ms", {}).keys())
    stages = {}
    for k in sorted(keys):
        vals = sorted(t["stages_ms"][k] for t in done if t["stages_ms"].get(k) is not None)
        if vals:
            stages[k] = {
                "n": len(vals), "avg": round(sum(vals) / len(vals), 1), "min": vals[0],
                "p50": _pct(vals, 0.5), "p95": _pct(vals, 0.95), "max": vals[-1],
            }
    outcomes = {}
    for t in _recent_turns:
        outcomes[t.get("outcome")] = outcomes.get(t.get("outcome"), 0) + 1
    return {"turns_total": len(_recent_turns), "outcomes": outcomes, "stages_ms": stages}


def tail_log(lines=300):
    try:
        with open(SERVER_LOG, "r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except FileNotFoundError:
        return ""


class TurnTrace:
    """Collects monotonic timestamps for one turn and renders a timeline + stage durations."""

    TIMELINE = [
        ("last_speech", "last speech chunk received from client"),
        ("trigger", "turn triggered (silence limit reached)"),
        ("stt_start", "STT started (worker thread dispatched)"),
        ("stt_end", "STT finished"),
        ("llm_start", "LLM request sent to Ollama"),
        ("llm_first_token", "LLM first token received"),
        ("tts_first_start", "first TTS synthesis started"),
        ("tts_first_end", "first TTS synthesis finished"),
        ("first_audio_sent", ">>> FIRST AUDIO SENT TO CLIENT <<<"),
        ("llm_stream_end", "LLM stream finished"),
        ("last_audio_sent", "last audio chunk sent"),
        ("turn_end", "turn finished"),
    ]

    def __init__(self, sid, turn):
        self.sid = sid
        self.turn = turn
        self.marks = {}
        self.stage = "pending"
        self.tokens = 0
        self.token_offsets_ms = []
        self.tts_calls = []
        self.stt = {}
        self.llm = {}
        self.audio_in_s = 0.0
        self.segment = {}
        self.wall_start = time.time()

    def mark(self, name, only_first=False):
        if only_first and name in self.marks:
            return
        self.marks[name] = time.perf_counter()

    def ms(self, a, b):
        if a in self.marks and b in self.marks:
            return round((self.marks[b] - self.marks[a]) * 1000, 1)
        return None

    def stages(self):
        tts_total = round(sum(c.get("total_ms", 0) for c in self.tts_calls), 1)
        stages = {
            "endpointing_wait": self.ms("last_speech", "trigger"),
            "stt": self.ms("stt_start", "stt_end"),
            "stt_to_llm_request": self.ms("stt_end", "llm_start"),
            "llm_ttft": self.ms("llm_start", "llm_first_token"),
            "phrase_buffer_wait": self.ms("llm_first_token", "tts_first_start"),
            "tts_first_synth": self.ms("tts_first_start", "tts_first_end"),
            "first_send": self.ms("tts_first_end", "first_audio_sent"),
            "trigger_to_first_audio": self.ms("trigger", "first_audio_sent"),
            "speech_end_to_first_audio": self.ms("last_speech", "first_audio_sent"),
            "turn_total_from_trigger": self.ms("trigger", "turn_end"),
            "tts_total_all_chunks": tts_total,
        }
        return {k: v for k, v in stages.items() if v is not None}

    def _base(self):
        return self.marks.get("last_speech") or self.marks.get("trigger")

    def timeline_text(self, header):
        base = self._base()
        lines = [header]
        prev = None
        for key, label in self.TIMELINE:
            if key not in self.marks:
                continue
            t = self.marks[key]
            off = (t - base) * 1000
            delta = f"(+{(t - prev) * 1000:7.1f}ms)" if prev is not None else "            "
            lines.append(f"    T+{off:8.1f}ms {delta}  {label}")
            prev = t
        return "\n".join(lines)

    def finalize(self, outcome, transcript="", reply="", extra=None):
        self.mark("turn_end")
        stages = self.stages()
        base = self._base()
        summary = {
            "ts": datetime.datetime.fromtimestamp(self.wall_start).isoformat(timespec="milliseconds"),
            "sid": self.sid,
            "turn": self.turn,
            "outcome": outcome,
            "user_text": transcript,
            "agent_text": reply,
            "audio_in_s": round(self.audio_in_s, 2),
            "segment": self.segment,
            "stage_at_end": self.stage,
            "stages_ms": stages,
            "stt": self.stt,
            "llm": self.llm,
            "llm_token_count": self.tokens,
            "llm_token_offsets_ms": self.token_offsets_ms,
            "tts_calls": self.tts_calls,
            "timeline_ms_from_last_speech": {
                k: round((self.marks[k] - base) * 1000, 1) for k, _ in self.TIMELINE if k in self.marks
            },
        }
        if extra:
            summary.update(extra)
        record_turn(summary)
        header = (
            f"  TURN {self.sid}/T{self.turn} {outcome.upper()} | "
            f"user={transcript!r} | agent={reply!r}"
        )
        text = self.timeline_text(header)
        text += "\n    ---- stage durations ----"
        for k, v in stages.items():
            text += f"\n    {k:<28}{v:>9.1f} ms"
        setup()
        _logger.info("\n" + text)
        return summary
