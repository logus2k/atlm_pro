"""MP1 — prepare the continued-pretraining corpus.

Builds a domain corpus of raw job-description text:
  - Djinni   -> train / val / test   (in-domain)
  - LinkedIn -> ood_test             (out-of-domain, cross-industry)

Continued pretraining needs only raw text — no labels, no teacher. Output is
one {"text": ...} JSON object per line.

  .venv_atlm_pro/bin/python src/prepare_corpus.py

Output: data/processed/mp1/{train,val,test,ood_test}.jsonl
"""
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DJINNI = ROOT / "data/jobs/djinni/train-00000-of-00001.parquet"
LINKEDIN = ROOT / "data/jobs/linkedin/postings.csv"
OUT = ROOT / "data/processed/mp1"

SEED = 42
MIN_CHARS = 200          # drop stub postings
MAX_CHARS = 8000         # drop outlier walls of text (keeps both sources comparable)
N_TRAIN = 12_000         # ~19 MB of text — well above the brief's 1-10 MB floor
N_VAL = 1_000
N_TEST = 1_000
N_OOD = 1_000


def clean(series):
    s = series.dropna().astype(str).str.strip()
    return s[s.str.len().between(MIN_CHARS, MAX_CHARS)]


def write(name, texts):
    path = OUT / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    mb = path.stat().st_size / 1e6
    chars = sum(len(t) for t in texts)
    print(f"  {name:9}: {len(texts):>6,} docs | {mb:6.1f} MB | "
          f"~{chars // 4:>9,} tokens (est.)")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # --- Djinni: in-domain train/val/test ---
    dj = pd.read_parquet(DJINNI, columns=["Long Description"])
    dj_texts = clean(dj["Long Description"]).tolist()
    rng.shuffle(dj_texts)
    need = N_TRAIN + N_VAL + N_TEST
    if len(dj_texts) < need:
        raise SystemExit(f"only {len(dj_texts):,} clean Djinni docs, need {need:,}")
    train = dj_texts[:N_TRAIN]
    val = dj_texts[N_TRAIN:N_TRAIN + N_VAL]
    test = dj_texts[N_TRAIN + N_VAL:need]

    # --- LinkedIn: out-of-domain test ---
    lk = pd.read_csv(LINKEDIN, usecols=["description"])
    lk_texts = clean(lk["description"]).drop_duplicates().tolist()
    rng.shuffle(lk_texts)
    ood = lk_texts[:N_OOD]

    print("MP1 continued-pretraining corpus:")
    write("train", train)
    write("val", val)
    write("test", test)
    write("ood_test", ood)
    print(f"\nwritten to {OUT}/  (seed={SEED}, train/val/test are disjoint Djinni docs)")


if __name__ == "__main__":
    main()
