# scripts/face_emotions_onnx.py — FER+ (ONNX) + MediaPipe, rapide & lissé
import os, json, argparse, cv2, numpy as np
import onnxruntime as ort
import mediapipe as mp

DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_FILE = "emotion-ferplus-8.onnx"

def ensure_model(model_dir=DEFAULT_MODEL_DIR, filename=DEFAULT_MODEL_FILE):
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, filename)
    if os.path.exists(local_path):
        return local_path
    # Tentative de téléchargement (facultatif)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="webai-community/models-bk", filename="emotion-ferplus-8.onnx")
        with open(path, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Modèle introuvable. Place {DEFAULT_MODEL_FILE} dans {DEFAULT_MODEL_DIR}/\nErreur: {e}"
        )

def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    arr = gray.astype(np.float32) / 255.0
    return arr[None, None, :, :]  # NCHW

def crop_face(frame_bgr, rbox, margin=0.1):
    H, W = frame_bgr.shape[:2]
    x = max(int((rbox.xmin - margin) * W), 0)
    y = max(int((rbox.ymin - margin) * H), 0)
    w = int((rbox.width  + 2*margin) * W)
    h = int((rbox.height + 2*margin) * H)
    x2 = min(x + w, W)
    y2 = min(y + h, H)
    if x2 <= x or y2 <= y:
        return None
    return frame_bgr[y:y2, x:x2].copy()

def smooth_timeline(tl, win=3):
    if not tl or win <= 1 or win % 2 == 0:
        return tl
    n, half, out = len(tl), win // 2, []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        labels = [tl[j]["emotion"] for j in range(a, b)]
        emo = max(set(labels), key=labels.count)
        out.append({**tl[i], "emotion": emo})
    return out

def analyze_emotions(video_path, sample_fps=2, smooth_win=3, max_timeline=1000, max_width=960):
    model_path = ensure_model()
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    labels = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps // max(1, sample_fps)), 1)

    detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    timeline, idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if (idx % step) != 0:
            idx += 1; continue

        # downscale rapide pour l’inférence (accélère MediaPipe)
        if max_width and frame.shape[1] > max_width:
            h = int(frame.shape[0] * (max_width / frame.shape[1]))
            frame = cv2.resize(frame, (max_width, h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        if result.detections:
            det = result.detections[0]
            rbox = det.location_data.relative_bounding_box
            face_img = crop_face(frame, rbox, margin=0.1)
            if face_img is not None and face_img.size > 0:
                inp = preprocess_face(face_img)
                logits = sess.run([out_name], {in_name: inp})[0][0]
                probs = softmax(logits)
                k = int(np.argmax(probs))
                t_sec = float(idx / fps)
                timeline.append({"t": round(t_sec, 2), "emotion": labels[k], "prob": round(float(probs[k]), 4)})

        idx += 1

    cap.release()

    if smooth_win and smooth_win >= 3 and smooth_win % 2 == 1:
        timeline = smooth_timeline(timeline, win=smooth_win)

    # Distribution / dominante
    counts = {k: 0 for k in labels}
    for it in timeline:
        if it["emotion"] in counts: counts[it["emotion"]] += 1
    total_samples = max(sum(counts.values()), 1)
    distribution = {k: round(counts[k] / total_samples, 3) for k in labels}
    dominant = max(distribution, key=distribution.get)

    # bornage de la timeline (évite des JSON énormes)
    if max_timeline and len(timeline) > max_timeline:
        timeline = timeline[:max_timeline]

    return {
        "model": "emotion-ferplus-onnx",
        "samples": total_samples,
        "distribution": distribution,
        "dominant_emotion": dominant,
        "timeline": timeline,
        "smoothed_win": smooth_win if (smooth_win and smooth_win >= 3 and smooth_win % 2 == 1) else 1,
        "sample_fps": sample_fps,
        "max_width": max_width
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Chemin de la vidéo")
    ap.add_argument("--sample-fps", type=int, default=2, help="1–2 recommandé pour 5–10min")
    ap.add_argument("--smooth-win", type=int, default=3, help="3 ou 5 conseillé")
    ap.add_argument("--max-timeline", type=int, default=1000, help="cap de la timeline JSON (0 = pas de cap)")
    ap.add_argument("--max-width", type=int, default=960, help="redimensionnement interne pour vitesse (px)")
    args = ap.parse_args()

    out = analyze_emotions(args.video, args.sample_fps, args.smooth_win, args.max_timeline, args.max_width)
    print(json.dumps(out, ensure_ascii=False))

