# Create an isolated environment. Run from the unpacked project directory.
# Device selects the official PyTorch wheel index, not the training algorithm.
param([ValidateSet("cpu", "cuda")][string]$Device = "cpu")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path ".venv/Scripts/python.exe")) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 x64 with the py launcher first." }
}
$VlmPython = Join-Path (Get-Location) ".venv/Scripts/python.exe"
$VlmIndex = if ($Device -eq "cuda") { "cu126" } else { "cpu" }
# Local version suffix forces a CPU/CUDA switch if .venv already exists.
& $VlmPython -m pip install "torch==2.8.0+$VlmIndex" "torchvision==0.23.0+$VlmIndex" --index-url "https://download.pytorch.org/whl/$VlmIndex"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $VlmPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $VlmPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency compatibility check failed." }
& $VlmPython -m phone_vlm doctor
if ($LASTEXITCODE -ne 0) { throw "Environment check failed. See the message above." }
Write-Host "Ready. Read docs/TRAINING.md before preparing the dataset."
