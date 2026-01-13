from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore

from ai_models import analyze_text

app = FastAPI(title="MindPulse API", version="0.1")

# CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev/hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


class AnalyzeRequest(BaseModel):
    text: str
    user_id: Optional[str] = "anonymous"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_text(req.text)

    doc = {
        "text": req.text,
        "user_id": req.user_id or "anonymous",
        "result": result,
        # Use Firestore server timestamp so sorting works reliably
        "created_at": firestore.SERVER_TIMESTAMP,
        # Also keep a readable string (optional)
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }

    db.collection("analyses").add(doc)
    return {"text": req.text, "user_id": doc["user_id"], **result}


@app.get("/history")
def history(user_id: str = "anonymous", limit: int = 10):
    # Fetch more than needed, then sort in Python (avoid Firestore index issues)
    docs = (
        db.collection("analyses")
        .where("user_id", "==", user_id)
        .limit(max(50, limit))
        .stream()
    )

    items = []
    for d in docs:
        x = d.to_dict() or {}
        x["id"] = d.id
        items.append(x)

    # Always convert created_at / created_at_iso into a comparable number (epoch seconds)
    def to_epoch_seconds(it):
        ts = it.get("created_at", None)

        # Firestore timestamp object (DatetimeWithNanoseconds) supports .timestamp()
        if ts is not None:
            try:
                return float(ts.timestamp())
            except Exception:
                pass

        # fallback: created_at_iso string
        iso = it.get("created_at_iso", "")
        if iso:
            try:
                # Python 3.10: fromisoformat works for "2026-01-12T06:01:25.378072+00:00"
                return datetime.fromisoformat(iso).timestamp()
            except Exception:
                pass

        return 0.0

    items.sort(key=to_epoch_seconds, reverse=True)
    items = items[:limit]

    # Make timestamps JSON friendly
    for it in items:
        ts = it.get("created_at")
        if ts is not None:
            try:
                it["created_at"] = ts.isoformat()
            except Exception:
                it["created_at"] = str(ts)

    return {"user_id": user_id, "count": len(items), "items": items}

