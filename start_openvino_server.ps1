Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " OpenVINO VLM Server - Intel Arc GPU Offload" -ForegroundColor Cyan
Write-Host " Hardware: Intel Core Ultra 7 258V (Arc 140V GPU / 16GB VRAM)" -ForegroundColor Cyan
Write-Host " Model   : Qwen2.5-VL-7B-Instruct-int4-ov" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Checking required packages..." -ForegroundColor Yellow
$check = python -c "import openvino_genai, huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing openvino-genai, huggingface_hub..." -ForegroundColor Green
    pip install openvino openvino-genai huggingface_hub pillow ultralytics
}

Write-Host "`n[2/2] Starting OpenVINO VLM Server (Port 8000 / GPU enabled)..." -ForegroundColor Yellow
Write-Host "Notice: First startup will download Qwen2.5-VL (~4.8GB) and compile for Intel Arc GPU." -ForegroundColor Gray

python openvino_server.py --host 0.0.0.0 --port 8000 --engine vlm --device GPU
