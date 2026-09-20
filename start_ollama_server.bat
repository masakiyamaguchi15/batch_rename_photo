@echo off
chcp 65001 > NUL
echo ========================================================
echo  Ollama LAN AI Server (Intel Core Ultra 7 258V)
echo  OLLAMA_HOST = 0.0.0.0:11434
echo ========================================================
set OLLAMA_HOST=0.0.0.0
ollama serve
pause
