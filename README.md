# atlm_pro Mini-Assignment 1: Continued Pretraining of SmolLM2-360M

Continued pretraining of the open [`SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) language model on a corpus of raw job-description text, adapting it to the job-postings domain. The training objective is plain next-token prediction, with no labels and no instructions.

The optimization experiment compares full fine-tuning against LoRA, with everything else held constant. Evaluation is by perplexity on an in-domain test set and an out-of-domain test set, plus sample generations before and after training.

This is Mini-Assignment 1 of a three-part project: Mini-Assignment 1 (continued pretraining), Mini-Assignment 2 (alignment), and a Final Project that builds a complete system.

## Results

Test-set perplexity, lower is better:

| model | in-domain (Djinni) | out-of-domain (LinkedIn) |
|---|---|---|
| base SmolLM2-360M | 16.37 | 17.80 |
| full fine-tuning | 13.12 (-20%) | 17.42 (-2%) |
| LoRA r=16 | 11.38 (-30%) | 18.27 (+3%) |

Continued pretraining gives strong in-domain adaptation, while out-of-domain perplexity barely moves: the adaptation is specific to IT job postings and does not transfer to cross-industry postings. LoRA adapts hardest in-domain while training only 2.3 percent of the parameters, though it also runs a higher learning rate (see the notebook's Section 5). The smaller SmolLM2-135M was also run as a comparison baseline; the notebook's Section 5.5 places the two model sizes side by side.

The full analysis, loss curves and generations are in [src/atlm_mp1_v4.ipynb](src/atlm_mp1_v4.ipynb).

## Setup

```bash
python -m venv .venv_atlm_pro
.venv_atlm_pro/bin/pip install -r requirements.txt
```

`requirements.lock.txt` holds the exact frozen versions used to produce the results above; install from it for an exact-match environment. Training requires an NVIDIA GPU with bf16 support (developed on an RTX 4090, 24 GB).

### Datasets

Place the raw datasets under `data/jobs/` (gitignored):

- Djinni (IT job descriptions; used for train, validation and the in-domain test): https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english, saved as `data/jobs/djinni/train-00000-of-00001.parquet`
- LinkedIn (cross-industry; used only for the out-of-domain test): https://www.kaggle.com/datasets/arshkon/linkedin-job-postings, saved under `data/jobs/linkedin/`

## How to run

The notebook is the primary deliverable. Open [src/atlm_mp1_v4.ipynb](src/atlm_mp1_v4.ipynb) and run it top to bottom. It is self-contained: it builds the corpus, runs the data analysis, trains both models, evaluates them, and produces every table and figure. On an RTX 4090 the two training runs take about one hour together.

The same steps are also available as standalone scripts:

```bash
# 1. Build the corpus into data/processed/mp1/
.venv_atlm_pro/bin/python src/prepare_corpus.py

# 2. The two continued-pretraining runs
.venv_atlm_pro/bin/python src/train.py --run mp1-360m --mode full
.venv_atlm_pro/bin/python src/train.py --run mp1-360m --mode lora

# 3. Evaluation into outputs/mp1-360m/eval.json
.venv_atlm_pro/bin/python src/evaluate_mp1.py --run mp1-360m
```

The trained model is a text completer, not a chat model: give it the start of a job posting and it continues. To try it:

```bash
.venv_atlm_pro/bin/python src/generate_mp1.py --model all --prompt "Senior Data Engineer"
.venv_atlm_pro/bin/python src/generate_mp1.py --model lora
```

## Reproducibility

- Random seed 42 is fixed for the corpus build, training and generation.
- Held constant across both training runs: 3 epochs, effective batch size 16 (micro-batch 4 with gradient accumulation 4), warmup ratio 0.03, weight decay 0.01, bf16, block size 1024. Only the learning rate and the LoRA settings differ between the two runs.
- Exact dependency versions are pinned in `requirements.lock.txt`.
- GPU training is not bit-for-bit deterministic, so perplexity can vary slightly between runs; the variation is far smaller than the differences the report draws conclusions from.

## Outputs

| Path | Contents |
|---|---|
| `data/processed/mp1/` | `train.jsonl`, `val.jsonl`, `test.jsonl` (in-domain), `ood_test.jsonl` (out-of-domain) |
| `outputs/mp1-360m/full/` | full fine-tuned model, `log_history.json`, `summary.json` |
| `outputs/mp1-360m/lora/` | LoRA adapter, `log_history.json`, `summary.json` |
| `outputs/mp1-360m/eval.json` | three-way perplexity and sample generations |

Each run writes to its own `outputs/<run>/` folder; the 135M comparison baseline is under `outputs/mp1-135m/`. `data/` and `outputs/` are gitignored, being large and regenerable.

## Layout

```
configs/    corpus and training configuration
src/        prepare_corpus.py, train.py, evaluate_mp1.py, generate_mp1.py
            atlm_mp1_v4.ipynb   the Mini-Assignment 1 notebook and report source
data/       jobs/ (raw, gitignored), processed/mp1/ (built corpus)
outputs/    trained models and evaluation, one folder per run
documents/  ma1_plan.md (project plan)
```
