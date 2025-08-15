# run_all.py — pipeline complet optimisé 5–10min
import os, sys, json, math, argparse, subprocess

def run_py(args):
    try:
        res = subprocess.run([sys.executable] + args, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("=== ERREUR sous-processus ===", file=sys.stderr)
        print("STDOUT:\n", e.stdout, file=sys.stderr)
        print("STDERR:\n", e.stderr, file=sys.stderr)
        raise

def _clamp(x, lo, hi): return max(lo, min(hi, float(x)))

WEIGHTS = {"verbal": 40.0, "paraverbal": 20.0, "nonverbal": 40.0}

def _score_wpm(wpm):
    if wpm is None: return 0.0
    if wpm <= 80 or wpm >= 200: return 0.0
    if wpm <= 120: return 12.0 * (wpm - 80) / 40.0
    if wpm <= 160: return 12.0
    return 12.0 * (200.0 - wpm) / 40.0

def _score_silence(silence_sec, duration_sec):
    if not duration_sec or silence_sec is None: return 0.0
    r = silence_sec / max(1e-6, duration_sec)
    if r <= 0.1: return 5.0
    if r >= 0.5: return 0.0
    return 5.0 * (0.5 - r) / 0.4

def _score_fillers(n_fillers, duration_sec):
    if duration_sec is None or n_fillers is None or duration_sec <= 0: return 0.0
    per_min = n_fillers / (duration_sec / 60.0)
    if per_min <= 1.0: return 3.0
    if per_min >= 8.0: return 0.0
    return 3.0 * (8.0 - per_min) / 7.0

def _score_eye_contact(ratio): return 20.0 * _clamp((ratio or 0.0), 0.0, 1.0)

def _score_nods(nods, duration_sec):
    if duration_sec is None or duration_sec <= 0 or nods is None: return 0.0
    npm = nods / (duration_sec / 60.0)
    if npm <= 0: return 0.0
    if npm <= 2: return 5.0 * (npm / 2.0)
    if npm <= 8: return 5.0 + 5.0 * ((npm - 2.0) / 6.0)
    if npm <= 16: return 10.0 - 5.0 * ((npm - 8.0) / 8.0)
    if npm <= 24: return 5.0 - 5.0 * ((npm - 16.0) / 8.0)
    return 0.0

def _score_emotion_variability(dist: dict, max_points: float = 10.0):
    if not dist: return 0.0
    import math
    p = [max(1e-9, float(v)) for v in dist.values()]
    s = sum(p)
    if s <= 0: return 0.0
    p = [x / s for x in p]
    H = -sum(x * math.log(x) for x in p)
    Hmax = math.log(len(p))
    return _clamp(max_points * (H / Hmax), 0.0, max_points)

def _baseline_verbal_from_length(text: str):
    if not text: return 0.0
    wc = max(0, len([w for w in text.split() if w.strip()]))
    if wc <= 50: return 0.0
    if wc >= 250: return 40.0
    return 40.0 * (wc - 50) / 200.0

def compute_scores(report: dict) -> dict:
    speech = report.get("speech", {}) or {}
    nonv   = report.get("nonverbal", {}) or {}
    duration = speech.get("duration_sec")
    paraverbal = _clamp(_score_wpm(speech.get("wpm")) +
                        _score_silence(speech.get("silence_sec"), duration) +
                        _score_fillers(speech.get("fillers"), duration), 0.0, WEIGHTS["paraverbal"])
    nonverbal = _score_eye_contact(nonv.get("eye_contact_ratio")) + _score_nods(nonv.get("nods"), duration)

    # mini-score émotion optionnel (entropie)
    if os.getenv("EMOTIONS_IN_SCORE", "0").lower() in ("1","true","yes"):
        emo_dist = (nonv.get("emotions") or {}).get("distribution", {})
        nonverbal += _score_emotion_variability(emo_dist, max_points=10.0)
    nonverbal = _clamp(nonverbal, 0.0, WEIGHTS["nonverbal"])

    qa = report.get("qa") or {}
    if "verbal_from_qa" in qa:
        verbal = qa["verbal_from_qa"]
    else:
        verbal = _baseline_verbal_from_length(report.get("transcript", "") or "")
    verbal = _clamp(verbal, 0.0, WEIGHTS["verbal"])

    total = round(verbal + paraverbal + nonverbal, 1)
    return {"verbal": round(verbal,1), "paraverbal": round(paraverbal,1), "nonverbal": round(nonverbal,1), "total": total}

def main(video, sample_fps=2, smooth_win=3, emo_max_timeline=1000, emo_max_width=960,
         stt_model="Systran/faster-whisper-base", stt_compute="int8", stt_beam=1, stt_lang=None):
    os.makedirs("outputs", exist_ok=True)

    # Stabilité OpenMP sous Windows
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # 1) Audio
    audio_path = "outputs/sample.wav"
    run_py(["scripts/extract_audio.py", video, audio_path])

    # 2) Transcription (rapide)
    transcript_path = "outputs/transcript.txt"
    run_py([
        "scripts/transcribe.py", audio_path, transcript_path,
        "--model", stt_model, "--device", "cpu",
        "--compute", stt_compute, "--beam", str(stt_beam)
    ] + (["--lang", stt_lang] if stt_lang else []))

    # 3) Paraverbal
    speech = json.loads(run_py(["scripts/speech_metrics.py", audio_path, transcript_path]))

    # 4) Texte (sentiment/résumé)
    text_metrics = json.loads(run_py(["scripts/text_analysis.py", transcript_path]))

    # 5) Non-verbal : émotions (rapide) + gaze/nods
    emotions = json.loads(run_py([
        "scripts/face_emotions_onnx.py", video,
        "--sample-fps", str(sample_fps),
        "--smooth-win", str(smooth_win),
        "--max-timeline", str(emo_max_timeline),
        "--max-width", str(emo_max_width),
    ]))
    gaze = eval(run_py(["scripts/gaze_nods.py", video]))

    report = {
        "transcript": open(transcript_path, "r", encoding="utf-8").read(),
        "speech": speech,
        "text": text_metrics,
        "nonverbal": {**gaze, "emotions": emotions},
    }

    # (optionnel) QA si tu veux plus tard : report["qa"] = {...}

    report["scores"] = compute_scores(report)

    out_path = "outputs/report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Chemin vers la vidéo")
    # Émotions
    ap.add_argument("--sample-fps", type=int, default=int(os.getenv("EMO_SAMPLE_FPS","2")))
    ap.add_argument("--smooth-win", type=int, default=int(os.getenv("EMO_SMOOTH_WIN","3")))
    ap.add_argument("--emo-max-timeline", type=int, default=int(os.getenv("EMO_MAX_TIMELINE","1000")))
    ap.add_argument("--emo-max-width", type=int, default=int(os.getenv("EMO_MAX_WIDTH","960")))
    # STT
    ap.add_argument("--stt-model", default=os.getenv("STT_MODEL","Systran/faster-whisper-base"))
    ap.add_argument("--stt-compute", default=os.getenv("STT_COMPUTE","int8"))
    ap.add_argument("--stt-beam", type=int, default=int(os.getenv("STT_BEAM","1")))
    ap.add_argument("--stt-lang", default=os.getenv("STT_LANG", None))
    args = ap.parse_args()

    main(args.video, args.sample_fps, args.smooth_win, args.emo_max_timeline, args.emo_max_width,
         args.stt_model, args.stt_compute, args.stt_beam, args.stt_lang)
