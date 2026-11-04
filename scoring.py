"""Shared model loading and scoring, used by both train.py and app.py.

Every scorer is a callable: scorer(list_of_texts) -> numpy array of toxicity
scores between 0 and 1.
"""
import os

import numpy as np

DEFAULT_BERT = "distilbert-base-uncased"


class BaselineScorer:
    """TF-IDF + logistic regression pipeline saved with joblib."""

    def __init__(self, path):
        import joblib

        self.pipe = joblib.load(path)

    def __call__(self, texts):
        return self.pipe.predict_proba(list(texts))[:, 1]

    def margin(self, texts):
        """Log-odds of toxic (does not saturate the way probabilities do)."""
        return self.pipe.decision_function(list(texts))


class BertScorer:
    """Fine-tuned (Distil)BERT sequence classifier loaded from a folder."""

    def __init__(self, model_dir, max_len=128, batch_size=128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir)
            .to(self.device)
            .eval()
        )
        self.max_len = max_len
        self.batch_size = batch_size

    def _logits(self, texts):
        torch = self.torch
        texts = list(texts)
        out = np.zeros((len(texts), 2), dtype=np.float32)
        if not texts:
            return out
        # sort by length so each batch needs little padding
        order = np.argsort([len(t) for t in texts])
        use_amp = self.device == "cuda"
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                idx = order[i : i + self.batch_size]
                enc = self.tok(
                    [texts[j] for j in idx],
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.model(**enc).logits
                out[idx] = logits.float().cpu().numpy()
        return out

    def __call__(self, texts):
        z = self._logits(texts)
        return 1.0 / (1.0 + np.exp(-(z[:, 1] - z[:, 0])))

    def margin(self, texts):
        """Log-odds of toxic (does not saturate the way probabilities do)."""
        z = self._logits(texts)
        return z[:, 1] - z[:, 0]


def load_scorers(artifacts_dir):
    """Load whichever of baseline / bert / bert_fixed exist in the folder."""
    scorers = {}
    base = os.path.join(artifacts_dir, "baseline.joblib")
    if os.path.exists(base):
        scorers["baseline"] = BaselineScorer(base)
    for name in ("bert", "bert_fixed"):
        d = os.path.join(artifacts_dir, name)
        if os.path.isdir(d):
            scorers[name] = BertScorer(d)
    return scorers


def explain(scorer, text, max_words=60):
    """Word-level explanation by occlusion.

    Remove one word at a time, re-score, and report how much the model's
    log-odds of "toxic" drop. Positive contribution = the word pushed the
    toxicity score up. Log-odds are used instead of probabilities because a
    probability near 1.0 hides the effect of any single word.
    Only the first `max_words` words are explained (the score uses the full text).
    Returns (probability_for_full_text, [{"word", "contribution"}, ...]).
    """
    tokens = text.split()
    prob = float(scorer([text])[0])
    n = min(len(tokens), max_words)
    if n == 0:
        return prob, []
    variants = [" ".join(tokens[:i] + tokens[i + 1 :]) for i in range(n)]
    margins = scorer.margin([text] + variants)
    base, rest = float(margins[0]), margins[1:]
    words = [{"word": tokens[i], "contribution": float(base - rest[i])} for i in range(n)]
    return prob, words
