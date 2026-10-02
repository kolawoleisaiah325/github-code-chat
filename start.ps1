$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath "$PSScriptRoot\.venv\Scripts\python.exe")) {
    throw 'Run ./setup.ps1 first to create the project environment.'
}
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\app.py" --server.address 127.0.0.1
