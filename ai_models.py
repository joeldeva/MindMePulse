from typing import Dict, Any
from transformers import pipeline

# =========================
# Models
# =========================
GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
PRIMARY_MH_MODEL = "mental/mental-health-roberta"
FALLBACK_MH_MODEL = "cardiffnlp/twitter-roberta-base-emotion"

# =========================
# Pipelines
# =========================
_emotion_pipe = pipeline(
    "text-classification",
    model=GOEMOTIONS_MODEL,
    return_all_scores=True
)

_sentiment_pipe = pipeline("sentiment-analysis")

try:
    _mh_pipe = pipeline("text-classification", model=PRIMARY_MH_MODEL)
except:
    _mh_pipe = pipeline("text-classification", model=FALLBACK_MH_MODEL)

# =========================
# Helpers
# =========================
def group_to_6(emotions: Dict[str, float]) -> Dict[str, float]:
    joy = ["joy", "love", "gratitude", "optimism", "amusement", "excitement"]
    sadness = ["sadness", "grief", "remorse", "disappointment"]
    anger = ["anger", "annoyance", "disgust"]
    anxiety = ["fear", "nervousness"]
    surprise = ["surprise", "confusion", "curiosity"]
    neutral = ["neutral"]

    out = {"joy":0,"sadness":0,"anger":0,"anxiety":0,"surprise_confusion":0,"neutral":0}

    for k,v in emotions.items():
        if k in joy: out["joy"] += v
        elif k in sadness: out["sadness"] += v
        elif k in anger: out["anger"] += v
        elif k in anxiety: out["anxiety"] += v
        elif k in surprise: out["surprise_confusion"] += v
        elif k in neutral: out["neutral"] += v

    s = sum(out.values())
    if s>0:
        for k in out:
            out[k] = out[k]/s

    return out

def compute_risk(e6, sent_label, sent_score):
    r = 0
    r += int(e6["sadness"]*40)
    r += int(e6["anxiety"]*35)
    r += int(e6["anger"]*15)
    if sent_label.lower()=="negative":
        r += int(sent_score*20)
    return min(100,max(0,r))

def detect_crisis(text):
    t = text.lower()
    high = ["suicide","kill myself","end my life","want to die","self harm","cut myself"]
    mid = ["hopeless","worthless","can't go on","give up","panic","overwhelmed"]

    for x in high:
        if x in t:
            return True,"high"
    for x in mid:
        if x in t:
            return True,"medium"
    return False,"low"

def safety_message(level):
    if level=="high":
        return "You may be in serious distress. Please contact local emergency services or a trusted person immediately."
    if level=="medium":
        return "You seem very overwhelmed. Talking to someone you trust or a mental health professional may help."
    return "If you feel worse, consider reaching out to someone you trust."

def generate_suggestions(e6, risk):
    if risk<35:
        return ["Take a short break","Drink water","Do 3 deep breaths"]
    if risk<70:
        return ["Try grounding exercise","Talk to someone you trust","Take a walk"]
    return ["Seek professional help","Reach out immediately","You are not alone"]

# =========================
# MAIN FUNCTION
# =========================
def analyze_text(text: str) -> Dict[str, Any]:

    # Emotions
    raw = _emotion_pipe(text)
    emo_list = raw[0] if isinstance(raw[0], list) else raw
    emotions_27 = {x["label"].lower(): float(x["score"]) for x in emo_list}
    emotions_6 = group_to_6(emotions_27)

    # Sentiment
    sent = _sentiment_pipe(text)[0]

    # Category
    try:
        mh = _mh_pipe(text)[0]
        mh_label = mh["label"]
        mh_conf = float(mh["score"])
    except:
        mh_label = "unknown"
        mh_conf = 0.0

    # Risk
    risk = compute_risk(emotions_6, sent["label"], float(sent["score"]))

    # Crisis
    crisis_flag, crisis_level = detect_crisis(text)
    safety = safety_message(crisis_level)

    # Suggestions
    suggestions = generate_suggestions(emotions_6, risk)

    return {
        "emotions_27": emotions_27,
        "emotions_6": emotions_6,
        "sentiment": sent,
        "risk_score": risk,
        "category": {"label": mh_label, "confidence": mh_conf},
        "crisis_flag": crisis_flag,
        "crisis_level": crisis_level,
        "safety_message": safety,
        "suggestions": suggestions
    }
