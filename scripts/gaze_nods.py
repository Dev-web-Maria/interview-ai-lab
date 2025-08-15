import cv2, sys, mediapipe as mp

def gaze_nods(video_path):
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    look_at_cam, total, nods, prev_nose_y = 0, 0, 0, None

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                cx = (lm[468].x + lm[473].x) / 2  # iris approx
                if 0.4 < cx < 0.6: look_at_cam += 1
                nose_y = lm[1].y
                if prev_nose_y is not None and abs(nose_y - prev_nose_y) > 0.015:
                    nods += 1
                prev_nose_y = nose_y
                total += 1
    cap.release()
    ratio = (look_at_cam / total) if total else 0.0
    return {"eye_contact_ratio": round(ratio,2), "nods": int(nods)}

if __name__ == "__main__":
    v = sys.argv[1]
    print(gaze_nods(v))
