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
