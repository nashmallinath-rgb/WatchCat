"""Responsible-AI toxicity project: training and evaluation.

Trains three models on the Jigsaw Unintended Bias in Toxicity Classification data:

  baseline    TF-IDF + logistic regression (counts words, ignores context)
  bert        DistilBERT fine-tuned on the data as it is
  bert_fixed  DistilBERT fine-tuned after adding extra *harmless* comments that
              mention gender (so the model sees that gender words alone are not
              a sign of toxicity)

Then it measures what each model does to comments that mention women or men,
sweeps the decision threshold (for the demo slider), and scores the handwritten
challenge set (survivor accounts vs subtle sexism).

Usage (Colab with a GPU):
    python train.py --data train.csv --challenge challenge_set.csv --out artifacts
Quick check without a GPU (baseline only):
    python train.py --data train.csv --challenge challenge_set.csv --out artifacts --skip_bert
Resume after a disconnect (skips models already saved in --out):
    python train.py --data train.csv --out artifacts --reuse
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from scoring import DEFAULT_BERT, BaselineScorer, BertScorer

SEED = 42
THRESH = 0.5
GENDER_COLS = ["female", "male"]
NEEDED = ["comment_text", "target"] + GENDER_COLS
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# --------------------------------------------------------------------- data
def load_data(path):
    df = pd.read_csv(path, usecols=lambda c: c in NEEDED)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing columns {missing}. Expected the Kaggle file "
            "train.csv from jigsaw-unintended-bias-in-toxicity-classification."
        )
    df = df.dropna(subset=["comment_text", "target"]).reset_index(drop=True)
    df["label"] = (df["target"] >= 0.5).astype(int)
    for c in GENDER_COLS:
        # identity columns are only filled for a subset of rows; blank = not annotated
        df[c] = df[c].fillna(0.0)
    df["female_mention"] = df["female"] >= 0.5
    df["male_mention"] = df["male"] >= 0.5
    df["gender_mention"] = df["female_mention"] | df["male_mention"]
    return df


def split_and_sample(df, n_train, n_test, gender_train_cap, gender_test_cap):
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(df))
    cut = int(len(df) * 0.9)
    train_pool, test_pool = df.iloc[order[:cut]], df.iloc[order[cut:]]

    # train: include plenty of gender-mention comments so the model really sees them
    gender = train_pool[train_pool["gender_mention"]]
    gender = gender.sample(min(len(gender), gender_train_cap), random_state=SEED)
    rest = train_pool.drop(gender.index)
    rest = rest.sample(min(len(rest), max(n_train - len(gender), 0)), random_state=SEED)
    train = pd.concat([gender, rest]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # test: a random slice for overall numbers, plus slices of gender-mention comments
    overall = test_pool.sample(min(n_test, len(test_pool)), random_state=SEED)
    parts = [overall.assign(set="overall")]
    for key in ("female_mention", "male_mention"):
        g = test_pool[test_pool[key]]
        g = g.sample(min(len(g), gender_test_cap), random_state=SEED)
        parts.append(g.assign(set=key))
    test = pd.concat(parts).reset_index(drop=True)
    return train, test


def rebalance(train, copies):
    """The 'fix': add extra copies of harmless comments that mention gender."""
    extra = train[(train["label"] == 0) & train["gender_mention"]]
    return (
        pd.concat([train] + [extra] * copies)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------- models
def train_baseline(train, path):
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True),
        LogisticRegression(C=4.0, max_iter=1000),
    )
    pipe.fit(train["comment_text"], train["label"])
    joblib.dump(pipe, path)


def train_bert(texts, labels, out_dir, bert_name, epochs, bs=32, lr=2e-5, max_len=128):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    print(f"  training {out_dir} on {dev}, {len(texts)} comments, {epochs} epoch(s)")
    tok = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2).to(dev)
    texts, labels = list(texts), np.asarray(labels)
    n = len(texts)
    total = epochs * ((n + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rng = np.random.RandomState(SEED)
    model.train()
    step, t0 = 0, time.time()
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, bs):
            idx = order[i : i + bs]
            enc = tok(
                [texts[j] for j in idx],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            y = torch.tensor(labels[idx], dtype=torch.long, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**enc, labels=y).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0 or step == total:
                print(f"    step {step}/{total}  loss {loss.item():.4f}  {time.time() - t0:.0f}s")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model
    if use_amp:
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ metrics
def frac(a):
    a = np.asarray(a)
    return float(a.mean()) if len(a) else None


def group_metrics(y, s, thresh=THRESH):
    y, s = np.asarray(y), np.asarray(s)
    auc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
    return {
        "n": int(len(y)),
        "toxic_rate": frac(y),
        "auc": auc,
        "false_positive_rate": frac(s[y == 0] >= thresh),  # harmless comments flagged
        "false_negative_rate": frac(s[y == 1] < thresh),  # toxic comments missed
    }


def challenge_summary(ch, scores, thresh=THRESH):
    out = {}
    for cat, g in ch.groupby("category"):
        s = scores[g.index.values]
        out[cat] = {
            "n": int(len(g)),
            "mean_score": float(s.mean()),
            "flag_rate": float((s >= thresh).mean()),
        }
    return out


def sweep_table(test, test_scores, ch, ch_scores):
    sets = {
        k: (g["label"].values, test_scores[g.index.values]) for k, g in test.groupby("set")
    }
    surv = ch_scores[ch["category"].values == "survivor"]
    sexist = ch_scores[ch["category"].values == "subtle_sexism"]
    rows = []
    for t in THRESHOLDS:
        t = float(t)
        row = {"threshold": t}
        for name, key in (("female", "female_mention"), ("male", "male_mention")):
            y, s = sets[key]
            row[f"{name}_mention_harmless_flagged"] = frac(s[y == 0] >= t)
        y, s = sets["overall"]
        row["overall_harmless_flagged"] = frac(s[y == 0] >= t)
        row["overall_toxic_missed"] = frac(s[y == 1] < t)
        row["survivor_flagged"] = frac(surv >= t)
        row["subtle_sexism_missed"] = frac(sexist < t)
        rows.append(row)
    return rows


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to Kaggle train.csv")
    ap.add_argument("--challenge", default="challenge_set.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--n_train", type=int, default=60000)
    ap.add_argument("--n_test", type=int, default=30000)
    ap.add_argument("--gender_train_cap", type=int, default=15000)
    ap.add_argument("--gender_test_cap", type=int, default=10000)
    ap.add_argument("--copies", type=int, default=3, help="extra copies of harmless gender comments")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bert_name", default=DEFAULT_BERT)
    ap.add_argument("--skip_bert", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="skip training for models already in --out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print("loading data...")
    df = load_data(args.data)
    print(f"  {len(df)} comments, {df['label'].mean():.1%} toxic, "
          f"{int(df['female_mention'].sum())} mention women, {int(df['male_mention'].sum())} mention men")
    train, test = split_and_sample(
        df, args.n_train, args.n_test, args.gender_train_cap, args.gender_test_cap
    )
    fixed = rebalance(train, args.copies)
    print(f"  train {len(train)} (fixed version {len(fixed)}), test {len(test)}")

    ch = pd.read_csv(args.challenge).reset_index(drop=True)

    scorers = {}
    base_path = os.path.join(args.out, "baseline.joblib")
    if not (args.reuse and os.path.exists(base_path)):
        print("training baseline...")
        train_baseline(train, base_path)
    scorers["baseline"] = BaselineScorer(base_path)

    if not args.skip_bert:
        for name, data in (("bert", train), ("bert_fixed", fixed)):
            d = os.path.join(args.out, name)
            if not (args.reuse and os.path.isdir(d)):
                print(f"training {name}...")
                train_bert(data["comment_text"], data["label"], d, args.bert_name, args.epochs)
            scorers[name] = BertScorer(d)

    results, sweeps = {}, {}
    ch_table = ch.copy()
    for name, scorer in scorers.items():
        print(f"scoring with {name}...")
        ts = scorer(test["comment_text"])
        cs = scorer(ch["comment_text"] if "comment_text" in ch else ch["text"])
        ch_table[name] = cs
        results[name] = {
            "by_set": {
                key: group_metrics(g["label"].values, ts[g.index.values])
                for key, g in test.groupby("set")
            },
            "challenge": challenge_summary(ch, cs),
        }
        sweeps[name] = sweep_table(test, ts, ch, cs)

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    json.dump(sweeps, open(os.path.join(args.out, "sweep.json"), "w"), indent=2)
    ch_table.to_csv(os.path.join(args.out, "challenge_scores.csv"), index=False)
    json.dump(ch_table.to_dict(orient="records"),
              open(os.path.join(args.out, "challenge_scores.json"), "w"), indent=2)

    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}%"

    print(f"\n=== summary at threshold {THRESH} ({time.time() - t0:.0f}s total) ===")
    print(f"{'model':<11}{'AUC':>7}  {'harmless flagged: women':>24}{'men':>8}  "
          f"{'survivors flagged':>18}{'sexism missed':>15}")
    for name, r in results.items():
        o, f, m = (r["by_set"][k] for k in ("overall", "female_mention", "male_mention"))
        c = r["challenge"]
        sexism_missed = 1 - c["subtle_sexism"]["flag_rate"] if "subtle_sexism" in c else None
        print(f"{name:<11}{(o['auc'] or 0):7.3f}  {pct(f['false_positive_rate']):>24}"
              f"{pct(m['false_positive_rate']):>8}  "
              f"{pct(c.get('survivor', {}).get('flag_rate')):>18}{pct(sexism_missed):>15}")
    print(f"\nsaved everything to {args.out}/")


if __name__ == "__main__":
    main()
