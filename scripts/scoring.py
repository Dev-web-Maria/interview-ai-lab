# scripts/scoring.py
from __future__ import annotations
import math

WEIGHTS = {
    "verbal": 40,
    "paraverbal": 20,
    "nonverbal": 40,
}

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _safe_div(a, b, default=0.0):
    return a / b if b else default

def _score_wpm(wpm: float) -> float:
    """
    12 points.
    Zone idéale: 110–170 WPM -> 12 pts.
    Tolérance: chute linéaire jusqu'à 0 pt à 70 et 230 WPM.
    """
    ideal_min, ideal_max = 110, 170
    hard_min, hard_max = 70, 230
    if ideal_min <= wpm <= ideal_max:
        return 12.0
    if wpm < ideal_min:
        return 12.0 * _clamp((wpm - hard_min) / (ideal_min - hard_min), 0.0, 1.0)
    else:
        return 12.0 * _clamp((hard_max - wpm) / (hard_max - ideal_max), 0.0, 1.0)

def _score_silence(silence_sec: float, duration_sec: float) -> float:
    """
    6 points.
    <=15% de silences -> 6 pts.
    >=40% -> 0 pt. Linéaire entre les deux.
    """
    r = _safe_div(silence_sec, duration_sec)
    if r <= 0.15:
        return 6.0
    if r >= 0.40:
        return 0.0
    # map 0.15..0.40 -> 6..0
    return 6.0 * (0.40 - r) / (0.40 - 0.15)

def _score_fillers(fillers: int, duration_sec: float) -> float:
    """
    2 points.
    <=2 fillers/min -> 2 pts.
    >=10 fillers/min -> 0 pt. Linéaire entre.
    """
    fpm = _safe_div(fillers, duration_sec / 60.0)
    if fpm <= 2.0:
        return 2.0
    if fpm >= 10.0:
        return 0.0
    return 2.0 * (10.0 - fpm) / (10.0 - 2.0)

def _score_eye_contact(eye_contact_ratio: float) -> float:
    """
    30 points.
    Saturation à 0.8 (80% de regard).
    """
    return 30.0 * _clamp(eye_contact_ratio / 0.8, 0.0, 1.0)

def _score_nods(nods: int, duration_sec: float) -> float:
    """
    10 points.
    Pic à ~9 hochements/min; 0 pt à 0 et 18 h/min (triangulaire).
    """
    npm = _safe_div(nods, duration_sec / 60.0)
    return 10.0 * _clamp(1.0 - abs(npm - 9.0) / 9.0, 0.0, 1.0)

def _score_summary(summary: str) -> float:
    """
    28 points.
    Proxie très simple: longueur utile du résumé (0..300+ chars).
    """
    n = len((summary or "").strip())
    return _clamp(28.0 * (n / 300.0), 0.0, 28.0)

def _score_sentiment(label: str, conf: float) -> float:
    """
    12 points.
    positive: de 0..12 pts (∝ confiance)
    neutral : ~3..12 pts
    negative: ~3..9 pts (plus la négativité est forte, plus ça baisse)
    """
    label = (label or "").lower()
    conf = _clamp(conf or 0.0, 0.0, 1.0)
    if label == "positive":
        base = 12.0 * conf
    elif label == "neutral":
        base = 3.0 + 9.0 * conf
    elif label == "negative":
        base = 3.0 + 6.0 * (1.0 - conf)
    else:
        base = 6.0  # inconnu
    return _clamp(base, 0.0, 12.0)

def compute_scores(report: dict) -> dict:
    """
    Calcule verbal / paraverbal / nonverbal / total (sur 100)
    à partir du report brut produit par le pipeline.
    """
    speech = report.get("speech", {})
    text = report.get("text", {})
    nonv = report.get("nonverbal", {})

    duration = float(speech.get("duration_sec") or 0.0)
    wpm = float(speech.get("wpm") or 0.0)
    silence = float(speech.get("silence_sec") or 0.0)
    fillers = int(speech.get("fillers") or 0)

    eye = float(nonv.get("eye_contact_ratio") or 0.0)
    nods = int(nonv.get("nods") or 0)

    sent = text.get("sentiment", {}) or {}
    sent_label = sent.get("label") or ""
    sent_score = float(sent.get("score") or 0.0)
    summary = (text.get("summary") or "").strip()

    # Sous-scores
    paraverbal = _score_wpm(wpm) + _score_silence(silence, duration) + _score_fillers(fillers, duration)
    nonverbal  = _score_eye_contact(eye) + _score_nods(nods, duration)
    verbal     = _score_summary(summary) + _score_sentiment(sent_label, sent_score)

    # Normalisation aux pondérations
    verbal = _clamp(verbal, 0.0, WEIGHTS["verbal"])
    paraverbal = _clamp(paraverbal, 0.0, WEIGHTS["paraverbal"])
    nonverbal = _clamp(nonverbal, 0.0, WEIGHTS["nonverbal"])

    total = verbal + paraverbal + nonverbal
    return {
        "verbal": round(verbal, 1),
        "paraverbal": round(paraverbal, 1),
        "nonverbal": round(nonverbal, 1),
        "total": round(total, 1),
        "details": {
            "wpm": _score_wpm(wpm),
            "silence": _score_silence(silence, duration),
            "fillers": _score_fillers(fillers, duration),
            "eye_contact": _score_eye_contact(eye),
            "nods": _score_nods(nods, duration),
            "summary_len": len(summary),
            "sentiment_label": sent_label,
            "sentiment_conf": sent_score,
        }
    }
