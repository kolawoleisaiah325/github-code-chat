param([switch]$WithModels)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install it and retry.' }
}
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($WithModels) {
    ollama pull nomic-embed-text
    if ($LASTEXITCODE -ne 0) { throw 'Embedding model download failed.' }
    ollama pull llama3.2
    if ($LASTEXITCODE -ne 0) { throw 'Chat model download failed.' }
}
Write-Output 'Ready. Run ./start.ps1. The bundled example needs no model downloads.'
