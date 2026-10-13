# setup_env.ps1 for Windows
Write-Host "=== [1/3] Ensuring Ollama is Running & Pulling Model ==="
try {
    Invoke-RestMethod -Uri http://127.0.0.1:11434 -ErrorAction Stop
} catch {
    Write-Host "Starting Ollama..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow
    Start-Sleep -Seconds 3
}

Write-Host "Pulling qwen2.5:1.5b..."
ollama pull qwen2.5:1.5b

Write-Host "=== [2/3] Downloading Kokoro-82M Weights ==="
if (!(Test-Path -Path "models")) {
    New-Item -ItemType Directory -Path "models"
}
if (!(Test-Path -Path "models/kokoro-v0_19.onnx")) {
    Write-Host "Downloading kokoro-v0_19.onnx..."
    Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" -OutFile "models/kokoro-v0_19.onnx"
}
if (!(Test-Path -Path "models/voices.bin")) {
    Write-Host "Downloading voices.bin..."
    Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin" -OutFile "models/voices.bin"
}

Write-Host "=== Environment Setup Complete ==="
