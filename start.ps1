$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Ambiente virtual .venv nao encontrado em $venvPython"
    exit 1
}

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  FocusGuard - Monitor Multimodal de Foco (Concursos)" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Dashboard local : http://127.0.0.1:8765" -ForegroundColor Green
Write-Host "  Falar com Coach : Ctrl + Shift + Space" -ForegroundColor Yellow
Write-Host "  Modo Privacidade: Ctrl + Shift + P" -ForegroundColor Yellow
Write-Host "  Encerrar        : Ctrl + C ou Bandeja -> Sair" -ForegroundColor DarkGray
Write-Host ""

& $venvPython -m focus_guard.app.main @args
