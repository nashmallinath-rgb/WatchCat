"""Local scoring API for the demo.

Run:  uvicorn app:app --port 8000
Env:  ARTIFACTS_DIR (default "artifacts"), UI_DIR (default "ui")

If a folder named ui/ exists next to this file, it is served at http://localhost:8000/
so the page and the API share one origin (no CORS or file:// headaches on stage).
"""
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scoring import explain, load_scorers

ARTIFACTS = os.environ.get("ARTIFACTS_DIR", "artifacts")
UI_DIR = os.environ.get("UI_DIR", "ui")
MAX_CHARS = 1000

app = FastAPI(title="Toxicity ethics demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SCORERS = load_scorers(ARTIFACTS)


class ScoreRequest(BaseModel):
    text: str
    model: str = "bert_fixed"
    threshold: float = 0.5
    explain: bool = True


class CompareRequest(BaseModel):
    text: str
    threshold: float = 0.5


def clean(text):
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")
    return text[:MAX_CHARS]


def read_json(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found in {ARTIFACTS}/ (run train.py first)")
    with open(path) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"models": list(SCORERS), "artifacts_dir": ARTIFACTS}


@app.post("/score")
def score(req: ScoreRequest):
    if req.model not in SCORERS:
        raise HTTPException(400, f"unknown model '{req.model}'; available: {list(SCORERS)}")
    text = clean(req.text)
    scorer = SCORERS[req.model]
    if req.explain:
        s, words = explain(scorer, text)
    else:
        s, words = float(scorer([text])[0]), []
    return {
        "model": req.model,
        "score": s,
        "threshold": req.threshold,
        "flagged": s >= req.threshold,
        "words": words,
    }


@app.post("/compare")
def compare(req: CompareRequest):
    """Score one text with every available model (for the side-by-side view)."""
    text = clean(req.text)
    out = {}
    for name, scorer in SCORERS.items():
        s = float(scorer([text])[0])
        out[name] = {"score": s, "flagged": s >= req.threshold}
    return {"text": text, "threshold": req.threshold, "models": out}


@app.get("/sweep")
def sweep():
    """Threshold sweep per model: feeds the slider counters."""
    return read_json("sweep.json")


@app.get("/results")
def results():
    return read_json("results.json")


@app.get("/challenge")
def challenge():
    """Pre-scored handwritten sentences (survivor accounts, subtle sexism, ...)."""
    return read_json("challenge_scores.json")


# keep this last: it catches everything else and serves the static page
if os.path.isdir(UI_DIR):
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
