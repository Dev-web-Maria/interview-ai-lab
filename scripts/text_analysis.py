import sys, json
from transformers import pipeline

# modèles légers pour CPU
SENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"

def analyze_text(text_path):
    text = open(text_path, "r", encoding="utf-8").read()
    sent_pipe = pipeline("sentiment-analysis", model=SENT_MODEL, truncation=True)
    sum_pipe  = pipeline("summarization", model=SUM_MODEL)

    sentiment = sent_pipe(text[:2000])[0] if text else {}
    summary = sum_pipe(text[:2500], max_length=120, min_length=50, do_sample=False)[0]["summary_text"] if text else ""
    return {"sentiment": sentiment, "summary": summary}

if __name__ == "__main__":
    p = sys.argv[1]
    print(json.dumps(analyze_text(p), ensure_ascii=False))
