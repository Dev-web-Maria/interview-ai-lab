import argparse, json, os
from faster_whisper import WhisperModel

def transcribe(audio_path, out_txt, model_id="Systran/faster-whisper-base",
               device="cpu", compute="int8", beam=1, lang=None, vad=True):
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)

    # Modèle optimisé CPU (INT8) — très bon compromis vitesse/qualité
    model = WhisperModel(model_id, device=device, compute_type=compute)

    segments, info = model.transcribe(
        audio_path,
        language=lang,                    # None = auto
        beam_size=beam,                   # 1 pour vitesse max, >1 pour qualité
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    text = "".join(s.text.strip() + " " for s in segments).strip()
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    # Retour d’info utile (STDOUT) si tu veux logger
    meta = {
        "language": info.language,
        "duration": info.duration,
        "model": model_id,
        "compute": compute,
        "beam": beam,
        "vad": vad
    }
    print(json.dumps(meta, ensure_ascii=False))
    return out_txt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Chemin WAV/MP3/M4A…")
    ap.add_argument("out_txt", help="Chemin du transcript .txt")
    ap.add_argument("--model", default="Systran/faster-whisper-base")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","auto"])
    ap.add_argument("--compute", default="int8", help="int8 | int8_float16 | float16 | float32")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--lang", default=None, help="ex: en, fr; None=auto")
    ap.add_argument("--no-vad", action="store_true", help="désactive le VAD")
    args = ap.parse_args()

    transcribe(
        args.audio, args.out_txt,
        model_id=args.model,
        device=args.device,
        compute=args.compute,
        beam=args.beam,
        lang=args.lang,
        vad=(not args.no_vad)
    )
