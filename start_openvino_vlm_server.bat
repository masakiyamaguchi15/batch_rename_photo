@echo off
chcp 65001 > NUL
echo ========================================================
echo  OpenVINO VLM (Vision LLM) Server - Intel Arc GPU Offload
echo  Hardware: Intel Core Ultra 7 258V (Arc 140V GPU / 16GB VRAM)
echo  Model   : Qwen2.5-VL-7B-Instruct-int4-ov (高性能マルチモーダルAI)
echo ========================================================

:: 必要なライブラリ (openvino-genai, huggingface_hub) の確認と自動導入
echo [1/3] 必要なライブラリを確認中...
python -c "import openvino_genai, huggingface_hub" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ライブラリをインストールしています (数分かかります)...
    pip install openvino openvino-genai huggingface_hub pillow ultralytics
)

echo.
echo [2/3] OpenVINO VLM サーバーを起動中 (モデルを Intel Arc GPU にオフロード)...
echo ※ 初回起動時は Hugging Face からモデル (約 4.8 GB) を自動ダウンロードし、
echo   GPU 向けコンパイルを行うため 2〜3 分程度お待ちください。
echo.

python openvino_server.py --host 0.0.0.0 --port 8000 --engine vlm --device GPU
pause
