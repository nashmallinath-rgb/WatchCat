# "Are you really safe on the internet?" — toxicity model + live demo API

Three comment classifiers trained on the Jigsaw Unintended Bias in Toxicity Classification data, plus a small local API your UI calls.

| model | what it is |
|---|---|
| `baseline` | TF-IDF + logistic regression. Counts words, ignores context. |
| `bert` | DistilBERT fine-tuned on the data as it is. Reads context. |
| `bert_fixed` | DistilBERT fine-tuned after adding extra *harmless* comments that mention women or men, so gender words alone stop looking like a warning sign. |

The demo story: how often does each model flag harmless comments that mention women or men, how does it treat survivor accounts versus subtle sexism (`challenge_set.csv`), and what does moving the threshold cost?

## Files

- `train.py` trains the three models, evaluates them, and writes everything the demo needs to `artifacts/`
- `scoring.py` shared model loading and word-level explanations
- `app.py` local FastAPI server (the live box, side-by-side view, slider data)
- `challenge_set.csv` 40 handwritten sentences: 12 survivor accounts, 12 subtle sexism, 8 neutral mentions of gender, 8 overt abuse
- `requirements.txt`

## Step 1: get the data (Kaggle)

1. Open the competition page `jigsaw-unintended-bias-in-toxicity-classification` on Kaggle and accept the rules (the download is blocked until you do).
2. Easiest: download `train.csv` in the browser (large, roughly a gigabyte unzipped).
3. Or in Colab: upload your `kaggle.json` (Kaggle → Settings → Create New Token), then

```
!pip -q install kaggle
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -f train.csv
!unzip -o train.csv.zip   # skip if it downloaded as plain train.csv
```

## Step 2: train on Colab

1. Runtime → Change runtime type → **T4 GPU**.
2. Upload `train.py`, `scoring.py`, `challenge_set.csv` (and `train.csv` unless you downloaded it in Colab).
3. Run:

```
!pip -q install transformers scikit-learn pandas joblib
!python train.py --data train.csv --challenge challenge_set.csv --out artifacts
```

4. Download the results: `!zip -r artifacts.zip artifacts`, then download `artifacts.zip` from the file panel.

Time: I estimate roughly 10 to 25 minutes per BERT model on a T4, but I could not run it on the real data, so treat that as a guess. If Colab disconnects, re-run with `--reuse` and it skips models already saved.

If it's too slow, shrink it: `--n_train 30000`. If you only want a quick sanity check, add `--skip_bert` (baseline only, a couple of minutes on CPU).

The script prints a summary table at the end:

- AUC (overall quality)
- share of *harmless* comments mentioning women / men that got flagged
- share of survivor accounts flagged
- share of subtle sexism missed

## Step 3: run the demo locally

On the laptop you'll present from:

```
pip install -r requirements.txt
# unzip artifacts.zip so the artifacts/ folder sits next to app.py
uvicorn app:app --port 8000
```

CPU is fine, a comment scores in well under a second. If you make a `ui/` folder next to `app.py`, your page is served at http://localhost:8000/ and calls the API on the same origin, so there's nothing to configure on stage and it works offline.

## API

| endpoint | what it does |
|---|---|
| `POST /score` `{"text": "...", "model": "bert_fixed", "threshold": 0.5, "explain": true}` | score, flagged yes/no, and per-word contributions. Positive = pushed toxicity up (log-odds units, so normalize before shading). |
| `POST /compare` `{"text": "...", "threshold": 0.5}` | the same text scored by every model, for the side-by-side view |
| `GET /sweep` | per model, per threshold: harmless comments flagged (women / men / overall), toxic comments missed, survivor accounts flagged, subtle sexism missed. Feeds the slider counters. |
| `GET /challenge` | the 40 handwritten sentences with every model's score, for the staged feed |
| `GET /results` | full evaluation numbers |
| `GET /health` | which models loaded |

## Honest limitations (put these on a slide)

- **Identity labels are partial.** Only a subset of the dataset's comments carry gender annotations. "Mentions women" means "annotated as mentioning women", and unannotated comments count as no mention.
- **The challenge set is 40 handwritten sentences.** It's good for illustrating failures, not for statistical claims. Say "examples", not "X% of survivors".
- **The fix is a simple rebalance** (extra copies of harmless gender-mention comments). It may help only a little, and it may make the model miss more subtle sexism. Report whatever you actually get, because that trade-off is the point of the project.
- **Labels are human judgments** of "toxic", so they carry the raters' own biases.
- The word highlighting uses occlusion (delete one word, re-score). It shows what the model reacts to, not a full explanation, and it can miss effects that depend on word pairs.
- Cite the dataset (Jigsaw / Conversation AI, via Kaggle) and credit your Lottie animations on the last slide.

## What was tested

I ran `train.py` and the API end to end on a small fake dataset with a tiny stand-in DistilBERT, so the code paths all work. I have **not** run it on the real Jigsaw data or the real `distilbert-base-uncased`, so no real results exist yet.
