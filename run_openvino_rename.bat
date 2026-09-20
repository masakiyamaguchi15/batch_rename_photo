@echo off
chcp 65001 > NUL
set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_ollama_rename.ps1" %*
