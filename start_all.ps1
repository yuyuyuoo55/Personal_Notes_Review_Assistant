$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSCommandPath
$logDirectory = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Test-LocalPort {
    param([int]$Port)

    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Start-LocalService {
    param(
        [int]$Port,
        [string]$Name,
        [string]$Command
    )

    if (Test-LocalPort -Port $Port) {
        Write-Host "$Name is already running on port $Port."
        return
    }

    $logFile = Join-Path $logDirectory "$Name.log"
    $cmdCommand = "$Command >> `"$logFile`" 2>&1"
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmdCommand -WorkingDirectory $projectRoot -WindowStyle Hidden
    Write-Host "Starting $Name..."
}

Start-LocalService -Port 8000 -Name 'backend' -Command 'uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000'
Start-LocalService -Port 8501 -Name 'frontend' -Command 'uv run streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true'

$frontendUrl = 'http://127.0.0.1:8501'
Write-Host "Project started. Opening: $frontendUrl"
Start-Process $frontendUrl
