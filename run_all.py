import os
import sys
import time
import signal
import subprocess
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath("."))

from app.database import init_db, get_batches, get_leads

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def check_url(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False

def print_banner():
    print("""
================================================================================
[*] JANBRIGHT WORLD SCHOOL, MOKILA
    Conversational Voice Admissions & Real-Time Workshop Reservation Ledger
================================================================================
    Curriculum: Cambridge Assessment International Education (CAIE)
    Weekend Workshops: Robotics (Ages 7-14), Archery (Ages 9-16), Swimming (Ages 6-15)
    Slots: Every Saturday at 10:30 AM & 3:00 PM
    Fee Policy: Strict Fee Deflection (Direct to on-campus meeting)
--------------------------------------------------------------------------------
    >> Audio Pipeline: Faster-Whisper tiny.en FP16 + Pinned Ollama Qwen2.5:1.5B
                      + Kokoro-82M CUDA TTS (< 400ms glass-to-glass)
================================================================================
""", flush=True)

def main():
    print_banner()

    # Step 1: Initialize Database & Verify Seed
    print("[1/4] Initializing school_enterprise.db relational ledger...")
    init_db()
    batches = get_batches()
    leads = get_leads()
    print(f"      Ledger ready: {len(batches)} batches loaded, {len(leads)} sample Mokila leads seeded.")

    # Step 2: Check Ollama Brain
    print("[2/4] Checking sovereign Ollama LLM service on port 11434...")
    ollama_ok = check_url("http://127.0.0.1:11434/api/tags")
    if ollama_ok:
        print("      Ollama is online and responsive.")
    else:
        print("      WARNING: Ollama service at http://127.0.0.1:11434 was not reachable.")
        print("      Please ensure 'ollama serve' is running with model 'qwen2.5:1.5b'.")

    venv_python_win = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe"))
    venv_python_nix = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin", "python"))
    if os.path.exists(venv_python_win):
        python_exe = venv_python_win
    elif os.path.exists(venv_python_nix):
        python_exe = venv_python_nix
    else:
        python_exe = sys.executable

    print(f"      Using Python environment: {python_exe}")

    # Step 3: Launch FastAPI Audio Backend
    print("[3/4] Launching FastAPI Sovereign Voice Server on http://127.0.0.1:8000...")
    fastapi_cmd = [
        python_exe, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--log-level", "warning"
    ]
    fastapi_proc = subprocess.Popen(fastapi_cmd)

    # Poll for backend readiness
    print("      Waiting for FastAPI server to initialize neural models...")
    for _ in range(30):
        if check_url("http://127.0.0.1:8000/health"):
            print("      FastAPI Audio Backend is LIVE and healthy!")
            break
        time.sleep(1.0)
    else:
        print("      FastAPI server taking longer to warm up; continuing...")

    # Step 4: Launch Streamlit Executive Dashboard
    print("[4/4] Launching Executive Streamlit Dashboard on http://127.0.0.1:8501...")
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--theme.base", "dark"
    ]
    streamlit_proc = subprocess.Popen(streamlit_cmd)

    print("\n" + "=" * 80)
    print("[*] PLATFORM SERVICES CONCURRENTLY OPERATIONAL")
    print("=" * 80)
    print("  [>] Executive Streamlit Dashboard : http://127.0.0.1:8501")
    print("  [>] FastAPI Backend & Health Docs  : http://127.0.0.1:8000/health")
    print("  [>] WebSocket Audio Gateway       : ws://127.0.0.1:8000/ws/call-stream")
    print("=" * 80)
    print("Press Ctrl+C to cleanly stop all platform services.\n", flush=True)

    def shutdown(signum=None, frame=None):
        print("\n[Shutting down Janbright Platform Services]...", flush=True)
        try:
            if streamlit_proc.poll() is None:
                streamlit_proc.terminate()
        except Exception:
            pass
        try:
            if fastapi_proc.poll() is None:
                fastapi_proc.terminate()
        except Exception:
            pass
        try:
            streamlit_proc.wait(timeout=3)
        except Exception:
            try:
                streamlit_proc.kill()
            except Exception:
                pass
        try:
            fastapi_proc.wait(timeout=3)
        except Exception:
            try:
                fastapi_proc.kill()
            except Exception:
                pass
        print("All processes successfully terminated. Goodbye.", flush=True)
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, shutdown)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, shutdown)
    except Exception:
        pass

    try:
        while True:
            f_code = fastapi_proc.poll()
            if f_code is not None:
                print(f"[ERROR]: FastAPI backend exited unexpectedly with code {f_code}", flush=True)
                shutdown()
                break
            s_code = streamlit_proc.poll()
            if s_code is not None:
                print(f"[ERROR]: Streamlit dashboard exited unexpectedly with code {s_code}", flush=True)
                shutdown()
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()
    finally:
        shutdown()

if __name__ == "__main__":
    main()
