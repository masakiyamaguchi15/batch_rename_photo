@echo off
chcp 65001 > NUL
echo ========================================================
echo  OpenVINO Dedicated LAN AI Server
echo  Hardware: Intel Core Ultra 7 258V (Intel Arc 140V GPU)
echo ========================================================

:: 必要なライブラリ (openvino, openvino-genai, huggingface_hub, ultralytics) の確認と自動導入
echo [1/2] 依存パッケージを確認中...
python -c "import openvino, openvino_genai, huggingface_hub, ultralytics" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 必要なパッケージ (openvino-genai, huggingface_hub 等) をインストールしています...
    pip install openvino openvino-genai huggingface_hub ultralytics pillow
)

echo [2/2] OpenVINO AI サーバーを起動中 (ポート 8000 / GPU 有効 / VLM モード)...
python openvino_server.py --host 0.0.0.0 --port 8000 --engine vlm --device GPU
pause
