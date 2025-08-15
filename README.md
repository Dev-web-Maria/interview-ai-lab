# interview-ai-lab (Phase 1)
POC d'analyse d'entretiens vidéo (verbal/paraverbal/non-verbal) en dehors du projet Django/React.

## Exécution rapide locale
1) Place une vidéo dans `data/sample.mp4`
2) Active le venv
3) `pip install -r requirements.txt`
4) `python run_all.py --video data/sample.mp4`
→ Résultat JSON dans `outputs/report.json`

## Quick start apres avoir cloner le repo (Windows)
1) git clone <URL_DU_REPO> && cd <repo>
2) powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1
3) Placez une vidéo dans `data/` (ex: data\sample.mp4)
4) .\.venv\Scripts\python.exe .\run_all.py --video "data\sample.mp4" --outdir "outputs\sample1"


---

# Interview AI Lab — Analyse d’entretiens vidéo (Django + React + IA locale)

**Objectif**
Analyser des entretiens vidéo “one-way” (5–10 min) et produire un **rapport JSON** combinant :

* **Verbal** : transcription (ASR)
* **Paraverbal** : débit (WPM), silences, fillers
* **Non verbal** : contact visuel, hochements de tête, **émotions faciales**
* **Scores** : agrégation simple pour lecture rapide par le recruteur

**Stack IA (100% local & gratuit, CPU)**

* **Transcription** : faster-whisper (CTranslate2, INT8)
* **Paraverbal** : librosa (analyse audio)
* **Non verbal** : MediaPipe (face), heuristiques (gaze/nods)
* **Émotions** : FER+ (ONNX, onnxruntime)
* **NLP** : Transformers (sentiment + résumé)

---

## 1) Prérequis

* **Windows 10/11** (ou Linux/Mac, adapter les chemins)
* **Python 3.11 x64**
* **FFmpeg** installé et présent dans le **PATH**

  * Windows : télécharger depuis gyan.dev, dézipper, ajouter `ffmpeg\bin` au `PATH`
* (Option intégration) **PostgreSQL**, **Node 18+** (front React), **Redis** (Celery)

---

## 2) Installation

```powershell
git clone <ton-repo>
cd interview-ai-lab

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> Les versions sont **epinglées** pour éviter les conflits (numpy/numba, protobuf/mediapipe, tokenizers/transformers…).

---

## 3) Arborescence

```
project/
├─ data/                        # tes vidéos d'entrée (non versionnées)
├─ models/                      # modèle FER+ ONNX (téléchargé auto si absent)
├─ outputs/                     # résultats par exécution (WAV, transcript, report.json)
├─ scripts/
│  ├─ extract_audio.py          # FFmpeg → WAV mono 16 kHz
│  ├─ transcribe.py             # faster-whisper (CPU INT8)
│  ├─ speech_metrics.py         # durée, WPM, silences, fillers
│  ├─ text_analysis.py          # sentiment + résumé court
│  ├─ face_emotions_onnx.py     # FER+ ONNX + MediaPipe, échantillonnage + lissage
│  └─ gaze_nods.py              # eye_contact_ratio + nods (heuristiques)
└─ run_all.py                   # orchestrateur → outputs/<uuid>/report.json
```

`.gitignore` recommandé :

```
.venv/
__pycache__/
*.pyc

outputs/*
!outputs/.gitkeep

data/*
!data/.gitkeep

models/*
!models/.gitkeep

*.mp4
*.wav
```

---

## 4) Lancer un test local (5–10 min vidéo)

1. Placer une vidéo d’entretien dans `data\sample.mp4`
2. Activer le venv et (option) variables pour OpenMP :

```powershell
.\.venv\Scripts\Activate.ps1
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"
```

3. Exécuter la pipeline (échantillonnage pour accélérer) :

```powershell
python run_all.py --video "data\sample.mp4" --outdir "outputs\sample1" ^
  --sample-fps 2 --smooth-win 3 --max-width 960 --max-timeline 1200
```

4. Consulter le résultat :

```powershell
type outputs\sample1\report.json
```

**Ce que contient `report.json`**

```json
{
  "transcript": "…",
  "speech": { "duration_sec": ..., "wpm": ..., "silence_sec": ..., "fillers": ... },
  "text": { "sentiment": {"label":"neutral","score":0.53}, "summary":"…" },
  "nonverbal": {
    "eye_contact_ratio": 0.62,
    "nods": 7,
    "emotions": {
      "dominant_emotion": "neutral",
      "distribution": {"neutral":0.52,"happiness":0.20,…},
      "timeline": [{"t":1.0,"e":"neutral"}, …]
    }
  },
  "scores": { "verbal": 40, "paraverbal": 20, "nonverbal": 40, "total": 100 }
}
```

> Paramètres utiles :
> `--sample-fps 1..2` (plus petit = plus rapide), `--max-width 960`, `--smooth-win 3/5`

---

## 5) Intégration Django + React (sans ligne de commande côté recruteur)

### Côté Django (extrait)

* **Modèle**

```python
# app/models.py
import uuid
from django.db import models

class Interview(models.Model):
    STATUS = [("PENDING","PENDING"),("RUNNING","RUNNING"),("DONE","DONE"),("ERROR","ERROR")]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    candidate_name = models.CharField(max_length=200, blank=True)
    video = models.FileField(upload_to="videos/%Y/%m/%d/")
    status = models.CharField(max_length=20, choices=STATUS, default="PENDING")
    report = models.JSONField(null=True, blank=True)
    duration_sec = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

* **Tâche Celery** (exécute `run_all.py`)

```python
# app/tasks.py
import os, sys, json, subprocess
from pathlib import Path
from celery import shared_task
from django.conf import settings
from .models import Interview

@shared_task(bind=True)
def run_analysis(self, interview_id):
    obj = Interview.objects.get(pk=interview_id)
    obj.status = "RUNNING"; obj.save(update_fields=["status"])

    outdir = Path(settings.MEDIA_ROOT) / "analysis" / str(obj.id)
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
    env.setdefault("OMP_NUM_THREADS","1")
    env.setdefault("HF_HOME", str(Path(settings.BASE_DIR) / ".hf_cache"))

    cmd = [sys.executable, str(Path(settings.BASE_DIR) / "run_all.py"),
           "--video", obj.video.path, "--outdir", str(outdir),
           "--sample-fps", "2", "--smooth-win", "3", "--max-width", "960"]
    p = subprocess.run(cmd, text=True, capture_output=True, env=env, cwd=settings.BASE_DIR)

    report_path = outdir / "report.json"
    if p.returncode != 0 or not report_path.exists():
        obj.status = "ERROR"
        obj.report = {"stderr": p.stderr}
        obj.save(update_fields=["status","report"])
        return

    with open(report_path, encoding="utf-8") as f:
        obj.report = json.load(f)
    obj.duration_sec = (obj.report.get("speech") or {}).get("duration_sec")
    obj.status = "DONE"
    obj.save(update_fields=["report","duration_sec","status"])
```

* **API DRF**

```python
# app/api.py
from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Interview
from .serializers import InterviewSerializer
from .tasks import run_analysis

class InterviewViewSet(ModelViewSet):
    queryset = Interview.objects.order_by("-created_at")
    serializer_class = InterviewSerializer

    @action(detail=True, methods=["post"])
    def analyze(self, request, pk=None):
        run_analysis.delay(pk)
        return Response({"ok": True})
```

Côté **React**, prévoir :

* un **uploader** (POST `/api/interviews/`),
* un bouton **Analyser** (POST `/api/interviews/:id/analyze/`),
* une page **Rapport** (GET `/api/interviews/:id/report/`).

> Le recruteur **ne tape aucune commande** : tout passe par l’UI ; les tâches tournent côté serveur.

---

## 6) Configuration

Variables d’environnement utiles (Windows) :

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"   # évite conflit OpenMP
$env:OMP_NUM_THREADS="1"           # limite les threads CPU
$env:HF_HOME="<chemin_cache_hf>"   # cache modèles partagé (optionnel)
```

Paramètres pipeline (CLI) :

* `--sample-fps` : frames analysées par seconde (1–2 recommandé CPU)
* `--smooth-win` : fenêtre de lissage émotions (3/5)
* `--max-width` : redimensionnement image (960 recommandé)
* `--max-timeline` : limite timeline émotions dans le JSON

---

## 7) Dépannage (FAQ)

* **FFmpeg non trouvé** → ajouter `<ffmpeg>\bin` au `PATH`
* **No such file** → vérifier le **chemin** de la vidéo (guillemets sous Windows)
* **OpenMP duplicate** (`libiomp5md.dll`) → `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`
* **Conflits numpy/numba/librosa** → garder `numpy==1.26.4` + `numba==0.58.1`
* **mediapipe / protobuf 6.x** → utiliser `protobuf==4.25.8`
* **Transformers / tokenizers** → `tokenizers==0.19.1` pour `transformers==4.41.2`
* **HF symlink warning (Windows)** → sans impact, cache un peu plus volumineux

---

## 8) Roadmap (idées)

* Génération **automatique** de rubriques d’évaluation par question (IA), avec **édition** par le recruteur
* Dashboard React (radar scores, timeline émotions, mots-clés du transcript)
* GPU optionnel (si dispo) pour accélérer STT
* Anonymisation vidéo (blur visage) si requis RGPD

---

