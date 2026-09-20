@echo off
chcp 65001 > NUL
echo ========================================================
echo  Ollama LAN AI Server (Intel Core Ultra 7 258V / Arc GPU)
echo ========================================================

:: 既存の Ollama プロセス（トレイアプリ含む）を一度停止してGPU設定を確実に反映
echo [1/3] 既存の Ollama プロセスを確認・再起動準備中...
taskkill /F /IM "ollama.exe" /IM "ollama app.exe" >nul 2>&1
timeout /t 2 /nobreak >nul

:: LAN公開設定 & CORS全許可
echo [2/3] LAN 公開設定 (0.0.0.0:11434)...
set OLLAMA_HOST=0.0.0.0:11434
set OLLAMA_ORIGINS=*

:: Intel Core Ultra / Arc GPU (Lunar Lake / Arc 140V) GPU オフロード設定
echo [3/3] Intel Arc GPU / Vulkan アクセラレーション設定...
set OLLAMA_VULKAN=1
set OLLAMA_INTEL_GPU=1
set OLLAMA_NUM_GPU=999
set ZES_ENABLE_SYSMAN=1
set SYCL_CACHE_PERSISTENT=1
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

:: モデルのアンロードを防ぐ（24時間保持）
set OLLAMA_KEEP_ALIVE=24h

:: GPU検出ログを出力
set OLLAMA_DEBUG=1

echo.
echo ========================================================
echo  サーバー起動中: http://0.0.0.0:11434
echo  ※ このウィンドウを開いたままにしておいてください
echo ========================================================
echo.

ollama serve
pause
