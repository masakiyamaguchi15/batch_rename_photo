Write-Host "================================================"
Write-Host " Ollama LAN Setup "
Write-Host "================================================"

Write-Host "[1/2] Setting OLLAMA_HOST=0.0.0.0..."
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
$env:OLLAMA_HOST = "0.0.0.0"

Write-Host "[2/2] Checking Ollama installation..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "  -> Ollama is installed."
} else {
    Write-Host "  -> Ollama is not in PATH."
}
Write-Host "Setup completed."
