@echo off
chcp 65001 > NUL
echo ========================================================
echo  OpenVINO Dedicated LAN AI Server
echo  Hardware: Intel Core Ultra 7 258V (Intel Arc 140V GPU)
echo ========================================================

:: 必要なライブラリ (openvino, ultralytics) の確認と自動導入
echo [1/2] 依存パッケージを確認中...
python -c "import openvino, ultralytics" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo OpenVINO および Ultralytics をインストールしています...
    pip install openvino ultralytics pillow
)

echo [2/2] OpenVINO AI サーバーを起動中 (ポート 8000 / GPU 有効)...
python openvino_server.py --host 0.0.0.0 --port 8000 --device GPU
pause
