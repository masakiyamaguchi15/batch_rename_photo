@echo off
chcp 65001 > NUL
echo ========================================================
echo  OpenVINO Dedicated LAN AI Server
echo  Hardware: Intel Core Ultra 7 258V - Intel Arc 140V GPU
echo ========================================================

echo [1/2] Checking dependencies...
python -c "import openvino, openvino_genai, huggingface_hub, ultralytics" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install openvino openvino-genai huggingface_hub ultralytics pillow
)

echo [2/2] Starting OpenVINO AI Server on port 8000 (VLM + GPU mode)...
python openvino_server.py --host 0.0.0.0 --port 8000 --engine vlm --device GPU
pause
