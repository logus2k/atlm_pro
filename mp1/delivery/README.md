# Mini-Assignment 1: Continued Pretraining of SmolLM2-360M

This bundle contains everything needed to reproduce the experiment and the
results in the accompanying report.

## Contents

- `atlm_ma1_groupc.ipynb`: the notebook, the primary code deliverable. It runs end to
  end: it builds the corpus, runs the data understanding, trains both models,
  evaluates them, and produces every table and figure in the report.
- `src/generate_mp1.py`: a small module that the notebook imports in
  Section 6.1 for interactive generation. The notebook expects it at this
  exact path.
- `requirements.txt`: the exact pinned versions of every Python package used
  to produce the reported numbers (the output of `pip freeze`).
- `atlm_ma1_report_groupc.pdf`: the technical report.

## Prerequisites

- Python 3.12.
- An NVIDIA CUDA GPU with bf16 support and roughly 12 GB of free VRAM. The
  notebook sets `DEVICE = "cuda"` explicitly; it will not run on CPU.
- Internet access on the first run. HuggingFace downloads two models the
  first time they are used: `HuggingFaceTB/SmolLM2-360M` (about 694 MB) and
  `sentence-transformers/all-MiniLM-L6-v2` (about 88 MB). Subsequent runs use
  the local cache.

## Environment setup

```
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are the exact versions used to
produce the report's numbers.

## Datasets

The notebook reads exactly two files. Both must be downloaded manually and
placed at the paths below. The notebook locates the data by searching for a
folder named `data` starting from the current working directory and walking
up, so `data/` must sit alongside the notebook or in one of its parents.

| Dataset | Source | Save as |
|---|---|---|
| Djinni (IT job descriptions) | https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english | `data/jobs/djinni/train-00000-of-00001.parquet` |
| LinkedIn (cross-industry) | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings | `data/jobs/linkedin/postings.csv` |

The total raw data needed is roughly 663 MB. The other files distributed with
those datasets (the `hf/` Djinni variant, the LinkedIn `companies/`, `jobs/`,
and `mappings/` sub-tables) are not used and can be ignored.

## Folder layout after setup

After downloading the datasets, the structure should be:

```
.
├── atlm_ma1_groupc.ipynb
├── atlm_ma1_report_groupc.pdf
├── README.md
├── requirements.txt
├── src/
│   └── generate_mp1.py
└── data/
    └── jobs/
        ├── djinni/
        │   └── train-00000-of-00001.parquet
        └── linkedin/
            └── postings.csv
```

## Running the notebook

Open `atlm_ma1_groupc.ipynb` in Jupyter or VS Code with the kernel pointing to the
virtual environment created above, and run all cells in order.

A complete run on a single RTX 4090 takes about one hour: 42 minutes for full
fine-tuning, 13 minutes for LoRA, and the rest for data understanding, corpus
construction and evaluation.

The notebook creates two folders during the run, alongside `data/`:

- `data/processed/mp1/`: the built corpus (training, validation, in-domain
  test, out-of-domain test).
- `outputs/mp1-360m/`: trained models and `eval.json`.

## Reproducibility

A single random seed (42) is fixed for corpus construction, model training,
and sampling. Exact library versions are pinned in `requirements.txt`.

One caveat: some CUDA operations are not bit-for-bit deterministic, so
perplexity values can vary slightly between runs even with the seed fixed.
The variation is well below the differences the report draws conclusions
from.

## Optional: model-size comparison in Section 5.5

Section 5.5 of the notebook compares the 360M results produced by this run
against a smaller 135M run. The 360M column comes from this notebook's run.
The 135M column is read from `outputs/mp1-135m/eval.json` if that file is
present; otherwise the cell prints a note and shows only the 360M column.
The full table as reported is in the technical report.
