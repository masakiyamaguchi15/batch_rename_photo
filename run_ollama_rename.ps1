# Ollama Photo Batch Rename Tool
$ScriptDir = $PSScriptRoot
$LocalVenv = "$ScriptDir\.venv\Scripts\Activate.ps1"
if (Test-Path $LocalVenv) {
    . $LocalVenv
}

# Ollama サーバーの動作確認 & 自動起動
$rawHost = $env:OLLAMA_HOST
if ([string]::IsNullOrEmpty($rawHost)) {
    $rawHost = "127.0.0.1:11434"
}
$ollamaHost = $rawHost.Replace("0.0.0.0", "127.0.0.1")
if (-not $ollamaHost.StartsWith("http")) {
    $ollamaHost = "http://$ollamaHost"
}
if (-not ($ollamaHost -like "*:11434*")) {
    $ollamaHost = "$ollamaHost:11434"
}

try {
    $null = Invoke-RestMethod -Uri "$ollamaHost/api/tags" -TimeoutSec 2 -ErrorAction Stop
} catch {
    Write-Host "[INFO] Ollama サーバーを起動中..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

$ScriptPath = "$ScriptDir\batch_rename_photos.py"
python $ScriptPath @args
