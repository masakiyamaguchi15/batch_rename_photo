Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " OpenVINO VLM Server - Intel Arc GPU Offload" -ForegroundColor Cyan
Write-Host " Hardware: Intel Core Ultra 7 258V (Arc 140V GPU / 16GB VRAM)" -ForegroundColor Cyan
Write-Host " Model   : Qwen2.5-VL-7B-Instruct-int4-ov" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] 必要なライブラリを確認中..." -ForegroundColor Yellow
$check = python -c "import openvino_genai, huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ライブラリ (openvino-genai, huggingface_hub) をインストールしています..." -ForegroundColor Green
    pip install openvino openvino-genai huggingface_hub pillow ultralytics
}

Write-Host "`n[2/2] OpenVINO VLM サーバーを起動中 (ポート 8000 / GPU 有効)..." -ForegroundColor Yellow
Write-Host "※ 初回起動時はモデル (約 4.8 GB) のダウンロードと GPU コンパイルで 2〜3分 かかります。" -ForegroundColor Gray

python openvino_server.py --host 0.0.0.0 --port 8000 --engine vlm --device GPU
