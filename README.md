# atlm_pro — MP1: Continued Pretraining of SmolLM2-135M

Continue the **pretraining** of the base [`SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
language model on a corpus of raw job-description text, adapting it to the
job-postings domain (next-token prediction — no labels, no queries).

**Optimization experiment:** full fine-tuning vs. LoRA, everything else held
constant. Evaluation is by perplexity on an in-domain test set and an
out-of-domain (OOD) test set, plus before/after sample generations.

This is **MP1** of a three-part project: MP1 continued pretraining → MP2
SFT + DPO alignment → Final capstone system.

## Results

Three-way **test-set perplexity** (lower = better):

| model              | in-domain (Djinni) | OOD (LinkedIn) |
|--------------------|--------------------|----------------|
| base SmolLM2-135M  | 20.43              | 23.97          |
| full fine-tuning   | 16.23  (−21%)      | 23.10  (−4%)   |
| LoRA r=16          | **13.64  (−33%)**  | 23.95  (−0%)   |

Strong in-domain adaptation; OOD perplexity barely moves — the adaptation is
domain-specific (IT job postings do not transfer to cross-industry postings).
LoRA adapted harder in-domain at ~3.5% of the trainable parameters (caveat: it
also ran a higher LR — see the notebook's Results section). Full analysis,
loss curves and generations are in [src/atlm_mp1_v2.ipynb](src/atlm_mp1_v2.ipynb).

## Setup

```bash
python -m venv .venv_atlm_pro
.venv_atlm_pro/bin/pip install -r requirements.txt   # or requirements.lock.txt for exact pins
```

Requires an NVIDIA GPU with bf16 support (developed on an RTX 4090, 24 GB).
`requirements.lock.txt` holds the exact frozen versions used to produce the
results above.

### Datasets

Place the raw datasets under `data/jobs/` (gitignored):

* **Djinni** (IT job descriptions, core train/val/test):
  https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english
  → `data/jobs/djinni/train-00000-of-00001.parquet`
* **LinkedIn** (cross-industry, OOD test only):
  https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
  → `data/jobs/linkedin/`

## How to run

The pipeline is three steps. All paths and hyperparameters live in
`configs/` — `data.yaml` (corpus build) and `train.yaml` (training).

```bash
# 1. Build the MP1 corpus -> data/processed/mp1/{train,val,test,ood_test}.jsonl
.venv_atlm_pro/bin/python src/prepare_corpus.py

# 2. Continued pretraining — the two experiment runs
.venv_atlm_pro/bin/python src/train.py --mode full
.venv_atlm_pro/bin/python src/train.py --mode lora

# 3. Three-way evaluation -> outputs/mp1_eval.json
.venv_atlm_pro/bin/python src/evaluate_mp1.py
```

Optionally, test the trained model interactively (it is a *text completer* —
give it the start of a job posting and it continues):

```bash
.venv_atlm_pro/bin/python src/generate_mp1.py --model all --prompt "Senior Data Engineer"
.venv_atlm_pro/bin/python src/generate_mp1.py --model lora            # REPL
```

The same logic, runnable cell-by-cell, is in
[src/atlm_mp1_v2.ipynb](src/atlm_mp1_v2.ipynb) (kernel: `atlm_pro`) — dataset
exploration (§1–3), continued pretraining (§4), results (§5) and interactive
testing (§6).

## Reproducibility

* **Seed:** `42` everywhere — corpus split (`configs/data.yaml`), training
  (`configs/train.yaml`, applied via `transformers.set_seed`), and generation.
* **Held constant** across both modes: 3 epochs, effective batch size 16
  (micro-batch 4 × grad-accum 4), warmup ratio 0.03, weight decay 0.01, bf16,
  `block_size` 1024. Only the `modes:` block of `configs/train.yaml` differs
  (learning rate; LoRA rank/alpha/dropout/target-modules).
* **Exact dependency versions:** `requirements.lock.txt`.

## Outputs / checkpoint locations

| Path                          | Contents                                                  |
|-------------------------------|-----------------------------------------------------------|
| `data/processed/mp1/`         | `train.jsonl` (12k) · `val.jsonl` (1k) · `test.jsonl` (1k, in-domain) · `ood_test.jsonl` (1k, LinkedIn) |
| `outputs/mp1-full/`           | full fine-tuned model + `log_history.json` · `summary.json` |
| `outputs/mp1-lora/`           | LoRA adapter + `log_history.json` · `summary.json`        |
| `outputs/mp1_eval.json`       | three-way perplexity + sample generations                 |

`data/` and `outputs/` are gitignored (large, regenerable).

## Layout

```
configs/    data.yaml (corpus), train.yaml (training)
src/        prepare_corpus.py, train.py, evaluate_mp1.py, generate_mp1.py
            atlm_mp1_v2.ipynb  — the MP1 notebook (EDA + training + results)
data/       jobs/ (raw, gitignored)  ·  processed/mp1/ (built corpus)
outputs/    trained checkpoints + evaluation
```
