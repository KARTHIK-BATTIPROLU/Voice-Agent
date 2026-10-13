# Sovereign Voice Agent: Real-Time Admissions & Workshop Reservation Engine 🎙️⚡

A 100% self-hosted, zero-cloud, sovereign full-duplex speech-to-speech voice orchestration pipeline and real-time enterprise reservation system designed for **Janbright World School, Mokila**.

Built with sub-400ms glass-to-glass latency, sub-100ms conversational barge-in (interruption handling), and a persistent SQLite relational ledger for Cambridge Assessment International Education (CAIE) admissions and weekend workshop registrations.

---

## 🌟 Architectural Highlights

1. **100% Self-Hosted & Zero API Keys**:
   - Zero reliance on external cloud APIs (no OpenAI, Cartesia, ElevenLabs, Deepgram, or Groq).
   - Operates fully offline on consumer hardware (tested on NVIDIA RTX 4050/4060 with < 4.5 GB VRAM footprint).

2. **Ultra-Low Latency Cascaded Pipeline**:
   - **Ear (STT)**: Faster-Whisper `tiny.en` (CUDA FP16) for instantaneous sub-120ms transcription.
   - **VAD**: Energy & Silero VAD tuned with 180ms silence trigger and immediate audio cut-off.
   - **Brain (LLM)**: Local Ollama instance pinned to `qwen2.5:1.5b` with streaming TTFT (Time to First Token) of ~150–200ms.
   - **Mouth (TTS)**: Kokoro-82M ONNX neural speech synthesizer running on CPU/CUDA yielding sub-100ms first audio chunk generation.

3. **Instant Barge-In & Interruption Handling**:
   - Sub-100ms cancellation: As soon as user speech is detected while the agent is speaking, outbound audio streaming cancels instantly (<15ms) and the turn state resets.

4. **Real-Time Enterprise Relational Ledger**:
   - SQLite relational database (`school_enterprise.db`) with atomic concurrency locks.
   - Manages workshop seat allocations (Robotics, Archery, Swimming) and parent admission leads.
   - Automated alternate slot suggestions when workshop batches reach full capacity.

5. **Dual User Interface & Observability**:
   - **Interactive Web Audio Console**: HTML5/Web Audio API frontend with live waveform visualizer, WebSocket binary PCM streaming, and live conversation transcript.
   - **Streamlit Command Center (`dashboard.py`)**: Real-time admissions ledger, workshop quota management, live latency metrics, and interactive database controls.

---

## 📐 System Architecture

```
                     +-------------------------------------------------------------+
                     |                         CLIENT TIER                         |
                     |  [Browser Web Audio API]  OR  [Python CLI Test Client]      |
                     +-------------------------------------------------------------+
                                      |                             ^
               16kHz PCM (512 samples)|                             | 16kHz PCM Audio &
               WebSocket Binary Frames|                             | JSON Transcripts
                                      v                             |
 +-------------------------------------------------------------------------------------------------+
 |                                       FASTAPI ASGI SERVER                                       |
 |                                                                                                 |
 | +---------------------------------------------------------------------------------------------+ |
 | |                         asyncio PRIMARY EVENT LOOP (CallAgent / Pipeline)                   | |
 | |                                                                                             | |
 | |  +---------------------+   Speech Detected   +--------------------+  Cancel   +-----------+ | |
 | |  | Energy & Silero VAD |-------------------->| Inbound Ring Buffer|---------->| Active    | | |
 | |  | CPU Inference (~1ms)|                     | (Accumulates PCM)  | (Barge-in)| TTS/LLM   | | |
 | |  +---------------------+                     +--------------------+           +-----------+ | |
 | |            |                                           |                          |         | |
 | |            | Silence >= 180ms                          | Trigger Turn             |         | |
 | |            v                                           v                          |         | |
 | |   [Turn Trigger Event] ----------------------> [Worker Pipeline]                  |         | |
 | +--------------------------------------------------------|--------------------------|---------+ |
 |                                                          |                          |           |
 | +--------------------------------------------------------v--------------------------v---------+ |
 | |                             ASYNC WORKER THREADS / LOCAL ENGINES                            | |
 | |                                                                                             | |
 | |  1. STT ENGINE:    Faster-Whisper tiny.en (CUDA FP16)             ~90 - 140 ms              | |
 | |  2. LLM ENGINE:    Ollama Qwen 2.5 1.5B (Streaming HTTP)          ~150 - 220 ms TTFT        | |
 | |  3. TOOL DISPATCH: Batches & Parent Leads SQLite Relational DB    ~5 - 15 ms                | |
 | |  4. TTS ENGINE:    Kokoro-82M ONNX Speech Synthesizer             ~80 - 150 ms              | |
 | +---------------------------------------------------------------------------------------------+ |
 |                                                                                                 |
 | +---------------------------------------------------------------------------------------------+ |
 | |                              TELEMETRY & DIAGNOSTICS SUBSYSTEM                              | |
 | |  - logs/server.log   (Formatted human-readable execution log with turn timelines)           | |
 | |  - logs/turns.jsonl  (Per-turn benchmark telemetry: stage latencies, TTFT, RTF, tokens/sec) | |
 | |  - logs/events.jsonl (Granular event streaming: VAD decisions, network jitter)              | |
 | +---------------------------------------------------------------------------------------------+ |
 +-------------------------------------------------------------------------------------------------+
```

---

## 📂 Project Directory Structure

```
Voice Agent/
├── ANTIGRAVITY_MASTER_SPEC.md         # Hardware constraints, latency SLAs & engineering requirements
├── ARCHITECTURE_AND_IMPLEMENTATION.md # Comprehensive latency profiling, component decomposition & tuning
├── README.md                          # Repository overview & operating instructions
├── requirements.txt                   # Python package dependencies
├── run_all.py                         # Unified orchestrator (Ollama check, DB init, Server + Dashboard)
├── run_server.bat                     # Quick-launch batch file for Windows
├── run_server.sh                      # Quick-launch shell script for Linux/WSL2
├── dashboard.py                       # Streamlit admissions & workshop operations command center
├── test_client.py                     # WebSocket audio test client & benchmark harness
├── app/
│   ├── __init__.py                    # Module export definitions
│   ├── call_agent.py                  # High-performance call agent with barge-in interruption logic
│   ├── config.py                      # Global latency parameters, sample rates & paths
│   ├── database.py                    # SQLite relational ledger (batches, parent leads, capacity)
│   ├── knowledge_base.py              # Context loader for CAIE curriculum & admissions FAQ
│   ├── llm_client.py                  # Async streaming Ollama client with TTFT tracking
│   ├── main.py                        # FastAPI application & WebSocket audio streaming endpoint
│   ├── pipeline.py                    # Voice pipeline coordinator with tool execution
│   ├── stt_engine.py                  # Faster-Whisper transcription wrapper
│   ├── telemetry.py                   # Stage-by-stage latency profiler & metric recorder
│   ├── tools.py                       # Structured action dispatcher (check seats, book batch)
│   ├── tts_engine.py                  # Kokoro-82M ONNX speech synthesis engine
│   ├── vad.py                         # Voice Activity Detection (Energy & Silero)
│   └── static/
│       └── index.html                 # Web Audio console with live waveform visualizer
├── knowledge/
│   ├── campus_faq.md                  # Campus transport, timings & security FAQ
│   ├── curriculum_and_admissions.md   # CAIE Cambridge curriculum details & fee deflection policy
│   ├── custom_rules.md                # Directives & live rules
│   ├── weekend_workshops.json         # Workshop eligibility & schedule catalog
│   └── sample_leads.xlsx              # Pre-seeded Mokila parent lead fixtures
├── models/
│   └── .gitkeep                       # Directory placeholder (weights auto-downloaded)
├── scripts/
│   ├── setup_env.ps1                  # Windows PowerShell automated model & environment setup
│   └── setup_env.sh                   # Linux/WSL2 automated setup script
└── tests/
    ├── test_barge_in.py               # Barge-in interruption cancellation verification
    ├── test_call_stream_ws.py         # Full-duplex WebSocket stream test
    ├── test_edge_cases.py             # Capacity overflow, boundary condition & error tests
    ├── test_janbright_platform.py     # End-to-end platform behavior validation
    ├── test_live_directive.py         # Admission fee deflection & workshop reservation policy tests
    └── test_pipeline_e2e.py           # End-to-end voice pipeline integration tests
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **OS**: Windows 11 / Ubuntu 22.04 / WSL2
- **Python**: 3.10+ (Recommended: Python 3.11)
- **GPU (Optional, Recommended)**: NVIDIA GPU with CUDA 12 support (RTX 3060/4050/4060 or higher)
- **Ollama**: Installed and serving locally (`http://127.0.0.1:11434`)

### 2. Environment Setup & Model Weights
Run the automated setup script to pull the Ollama LLM and download the Kokoro-82M weights:

**Windows PowerShell:**
```powershell
.\scripts\setup_env.ps1
```

**Linux / WSL2:**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch Services
Start the full stack (FastAPI server on port 8000 + Streamlit dashboard on port 8501):

```bash
python run_all.py
```

Or run individually:
- **FastAPI Voice Server**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **Streamlit Command Center**: `streamlit run dashboard.py --server.port 8501`

### 4. Access Interfaces
- **Web Voice Console**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Operations Dashboard**: Open [http://localhost:8501](http://localhost:8501) for live analytics.

---

## 🧪 Testing & Validation

Run the automated test suite covering all operational aspects:

```bash
pytest tests/ -v
```

Test specific subsystems:
- **Barge-in cancellation**: `pytest tests/test_barge_in.py -v`
- **Admissions policy & fee deflection**: `pytest tests/test_live_directive.py -v`
- **Capacity overflow & alternative booking**: `pytest tests/test_edge_cases.py -v`
- **WebSocket audio roundtrip**: `pytest tests/test_call_stream_ws.py -v`

---

## 📄 License
MIT License. Open for educational and enterprise deployment.
