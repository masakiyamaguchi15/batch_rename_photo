Write-Host "================================================"
Write-Host " Ollama LAN & Intel Arc GPU Setup "
Write-Host "================================================"

Write-Host "[1/3] Setting OLLAMA_HOST and Intel GPU Environment Variables..."
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_ORIGINS", "*", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_VULKAN", "1", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_INTEL_GPU", "1", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", "999", "User")
[System.Environment]::SetEnvironmentVariable("ZES_ENABLE_SYSMAN", "1", "User")
[System.Environment]::SetEnvironmentVariable("SYCL_CACHE_PERSISTENT", "1", "User")

Write-Host "[2/3] Checking Ollama installation..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "  -> Ollama is installed."
} else {
    Write-Host "  -> Ollama is not in PATH."
}

Write-Host "[3/3] Stopping background Ollama to apply new settings..."
Stop-Process -Name "ollama", "ollama app" -Force -ErrorAction SilentlyContinue

Write-Host "Setup completed. Please launch start_ollama_server.bat."
