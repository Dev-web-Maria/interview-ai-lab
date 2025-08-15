# scripts/setup-dev.ps1
# Setup dev Windows (PowerShell) pour interview-ai-lab
# - Crée .venv, installe requirements
# - Vérifie FFmpeg
# - Prépare dossiers + cache HF
# - Pré-télécharge Whisper (faster-whisper small int8)
# - Télécharge le modèle FER+ ONNX via scripts/face_emotions_onnx.py
# - Smoke test imports

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

# Affichage UTF-8 (évite PrÃ©...)
try {
  chcp 65001 | Out-Null
  [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
} catch {}

function Info([string]$msg) { Write-Host $msg -ForegroundColor Cyan }
function Step([string]$msg) { Write-Host ("• " + $msg) -ForegroundColor Green }
function Warn([string]$msg) { Write-Host ("! " + $msg) -ForegroundColor Yellow }
function Fail([string]$msg) { Write-Host ("× " + $msg) -ForegroundColor Red; exit 1 }

# Helper : exécuter un bout de Python en écrivant un .py temporaire (évite les soucis de quoting)
function Run-PyCode {
  param(
    [Parameter(Mandatory)] [string]$Code,
    [Parameter(Mandatory)] [string]$Label,
    [string]$PythonExe = $null
  )
  $tmp = Join-Path $env:TEMP ("ialab_" + [guid]::NewGuid().ToString() + ".py")
  Set-Content -Path $tmp -Value $Code -Encoding UTF8
  try {
    & $PythonExe $tmp
    if ($LASTEXITCODE -ne 0) { Fail $Label }
    else { Step "$Label OK" }
  } catch {
    Fail $Label
  } finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue | Out-Null
  }
}

# Racine repo
$Root = (Get-Location).Path
Info ("Repo: " + $Root)

# Dossiers
Info "Préparation des dossiers"
$dirs = @("data","models","outputs",".hf_cache")
foreach ($d in $dirs) {
  $p = Join-Path $Root $d
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
  Step "OK $d"
  # .gitkeep si absent
  $gk = Join-Path $p ".gitkeep"
  if (-not (Test-Path $gk)) { New-Item -ItemType File -Path $gk | Out-Null }
}

# FFmpeg
Info "Vérification FFmpeg"
$ff = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if ($ff) {
  Step ("FFmpeg: " + $ff.Path)
} else {
  Warn "FFmpeg absent dans le PATH. Télécharge: https://www.gyan.dev/ffmpeg/builds/ et ajoute ffmpeg\bin au PATH."
}

# Python 3.11 + venv
Info "Recherche Python 3.11"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
$venvExists = Test-Path $venvPy
if ($venvExists) {
  Step "venv déjà présent"
} else {
  # essai avec py -3.11 sinon python
  $created = $false
  try {
    & py -3.11 -V 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
      & py -3.11 -m venv ".\.venv"
      $created = $LASTEXITCODE -eq 0
    }
  } catch {}
  if (-not $created) {
    try {
      & python -V 2>$null | Out-Null
      & python -m venv ".\.venv"
      $created = $LASTEXITCODE -eq 0
    } catch {}
  }
  if (-not $created) { Fail "Impossible de créer le venv. Installe Python 3.11 (64-bit)." }
  Step "venv créé"
}
$PyExe = $venvPy
Step "Python OK → $PyExe"

# pip/setuptools/wheel
Info "Mise à jour pip/setuptools/wheel"
& $PyExe -m pip install -U pip setuptools wheel
Step "pip/setuptools/wheel OK"

# Install requirements (idempotent)
Info "Installation des dépendances"
$req = Join-Path $Root "requirements.txt"
if (-not (Test-Path $req)) { Fail "requirements.txt introuvable" }
& $PyExe -m pip install -r $req
Step "requirements OK"

# Variables d'environnement (session courante)
Info "Variables d'environnement (session)"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"
$env:HF_HOME = Join-Path $Root ".hf_cache"
Step "KMP_DUPLICATE_LIB_OK=TRUE"
Step "OMP_NUM_THREADS=1"
Step ("HF_HOME=" + $env:HF_HOME)

# Pré-téléchargements modèles
Info "Pré-téléchargement modèles (Whisper small INT8)"
Run-PyCode -PythonExe $PyExe -Label "Pré-download Whisper" -Code @'
from faster_whisper import WhisperModel
WhisperModel("Systran/faster-whisper-small", device="cpu", compute_type="int8")
print("OK Whisper cached")
'@

Info "Téléchargement FER+ ONNX (si manquant)"
Run-PyCode -PythonExe $PyExe -Label "Pré-download FER+" -Code @'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
try:
    import face_emotions_onnx as f
    f.ensure_model()
    print("OK FER+ model")
except Exception as e:
    print("WARN ensure_model():", e)
'@

# Smoke test imports (rapide)
Info "Smoke test imports"
Run-PyCode -PythonExe $PyExe -Label "Smoke test imports" -Code @'
import ffmpeg, av, numpy, onnxruntime, cv2, librosa
from faster_whisper import WhisperModel
print("OK imports")
'@

# Rappel d'usage
Info "`nTout est prêt ✅"
Write-Host "Prochaines étapes :" -ForegroundColor Cyan
Write-Host "1) Place une vidéo dans data\ (ex: data\sample.mp4)" -ForegroundColor Gray
Write-Host "2) Lance l'analyse :" -ForegroundColor Gray
Write-Host "   .\.venv\Scripts\python.exe .\run_all.py --video `"data\sample.mp4`" --outdir `"outputs\sample1`" --sample-fps 2 --smooth-win 3 --max-width 960" -ForegroundColor DarkGray
Write-Host "3) Résultats dans outputs\sample1\ (report.json, transcript.txt, etc.)" -ForegroundColor Gray
