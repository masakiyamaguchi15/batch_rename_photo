# OpenVINO -> Ollama Migration Photo Batch Rename Tool (PowerShell)
Write-Host "[INFO] AI Engine has been updated to Ollama." -ForegroundColor Cyan
$ScriptDir = $PSScriptRoot
& "$ScriptDir\run_ollama_rename.ps1" @args
