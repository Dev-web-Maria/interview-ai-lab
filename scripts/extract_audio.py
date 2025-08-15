import os, sys, ffmpeg

def extract_audio(video_path, out_wav, sr=16000):
    ffmpeg.input(video_path).output(out_wav, ac=1, ar=sr).overwrite_output().run()
    return out_wav

if __name__ == "__main__":
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/sample.wav"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(extract_audio(video, out))
