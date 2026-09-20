@echo off
chcp 65001 > NUL
echo ========================================================
echo  OpenVINO VLM Server - Intel Arc GPU Offload
echo  Hardware: Intel Core Ultra 7 258V - Arc 140V GPU - 16GB VRAM
echo  Model   : Qwen2.5-VL-7B-Instruct-int4-ov
echo ========================================================

echo [1/2] Checking required packages...
python -c "import openvino_genai, huggingface_hub" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing openvino-genai, huggingface_hub...
    pip install openvino openvino-genai huggingface_hub pillow ultralytics
)

echo [2/2] Starting OpenVINO VLM Server on port 8000 (GPU mode)...
python openvino_server.py --host 0.0.0.0 --port 8000 --engine vlm --device GPU
pause
