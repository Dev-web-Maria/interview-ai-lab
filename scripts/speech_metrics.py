import sys, json, re, librosa
from pathlib import Path

def speech_metrics(audio_path, transcript_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration_sec = len(y) / sr
    # Silences (approx) : durée non-parlée
    intervals = librosa.effects.split(y, top_db=30)  # segments parlés
    speech = sum((e - s) for s, e in intervals) / sr
    silence = max(0.0, duration_sec - speech)

    text = Path(transcript_path).read_text(encoding="utf-8")
    words = len(re.findall(r"\w+", text))
    wpm = words / (duration_sec / 60) if duration_sec > 0 else 0.0
    fillers = len(re.findall(r"\b(euh+|heu+|mmm+|bah|ben)\b", text.lower()))
    return {"duration_sec": round(duration_sec,2), "wpm": round(wpm,1), "silence_sec": round(silence,2), "fillers": fillers}

if __name__ == "__main__":
    audio, transcript = sys.argv[1], sys.argv[2]
    print(json.dumps(speech_metrics(audio, transcript), ensure_ascii=False))
