# Mini-Assignment 1 - Implementation Report

This document records how Mini-Assignment 1 (continued pretraining) was built: the development steps in the order they happened, what each step implemented, the most relevant code that carries the work, and the reasoning behind the options that were chosen. It is a development log, not the academic report. The academic report is the notebook `src/atlm_mp1_v4.ipynb`, which doubles as the reproducible code and as the source exported to the Word deliverable.

## 1. Goal and shape of the work

The brief asks for an open-source pretrained model under 1B parameters, continued-pretrained on a chosen domain (1 to 10 MB of text), with at least one training optimization explored through controlled runs, on the HuggingFace Transformers stack, and reported with loss curves, perplexity, and before/after generations.

The domain chosen was IT job postings. The model chosen was the SmolLM2 family. The controlled experiment was full fine-tuning versus LoRA, later widened with a second axis (model size: 135M versus 360M).

The implementation is split into four standalone scripts plus one notebook:

- `src/prepare_corpus.py` - builds the training corpus from raw datasets.
- `src/train.py` - continued pretraining, with the full-versus-LoRA knob.
- `src/evaluate_mp1.py` - perplexity and greedy generations for base/full/lora.
- `src/generate_mp1.py` - interactive generation (one-shot, compare, or REPL).
- `src/atlm_mp1_v4.ipynb` - the narrative, the analysis, and the report.

The training knobs and the experiment matrix live in `configs/train.yaml`, read by `train.py` at startup; the data-cleaning constants are hardcoded at the top of `src/prepare_corpus.py`. Splitting the training configuration out of code is what lets the full and LoRA runs differ in exactly one place. (An older `configs/data.yaml` sketch exists in the repo but is not wired in: `prepare_corpus.py` does not import yaml, and the constants in the script are what actually ran.)

## 2. Environment and reproducibility

Everything runs in a project virtual environment, `.venv_atlm_pro` (Python 3.12), on WSL2 (Ubuntu 24.04 on Windows 11). Two consumer GPUs were used during development, NVIDIA RTX 5060 Ti (16 GB) and NVIDIA RTX 4090 (24 GB); the 4090 carried the canonical runs whose numbers are reported here. Exact package versions are pinned in `requirements.lock.txt`.

A fixed seed of 42 is used everywhere it matters: the corpus shuffle, training (`set_seed`), and sampling. GPU training is not bit-for-bit deterministic even with a fixed seed, so figures vary slightly between runs; in practice the perplexity numbers reproduced exactly across reruns of the 360M experiment.

## 3. Step 1 - Data acquisition and understanding

Two public datasets were used:

- Djinni: roughly 142k IT job descriptions (a parquet file, the `Long Description` column). This is the in-domain source.
- LinkedIn: roughly 124k cross-industry postings (a CSV, the `description` column). This is the out-of-domain source, used only for testing, never for training.

The notebook's Data Understanding phase (Section 2) profiles both before any corpus is built: schema and missing values, text-length distributions, keyword and job-family distributions, lexical diversity, a tokenisation analysis of how the SmolLM2 tokenizer fragments technical terms, and a semantic-embedding analysis (sentence-transformers plus UMAP) that projects postings into 2D to show the cluster structure. The point of this phase is to justify the dataset choice with evidence rather than assertion, and to confirm the text is prompt-ready (long enough, clean enough, varied enough).

The decision that came out of this phase: Djinni is the cleaner, more consistently structured, more domain-focused source, so it carries train, val, and test; LinkedIn, being broader and noisier, is the natural out-of-domain probe.

## 4. Step 2 - Corpus construction

`src/prepare_corpus.py` turns the two raw datasets into four line-delimited JSON files, one `{"text": ...}` object per line. Continued pretraining needs only raw text, so there are no labels and no teacher at this stage.

The cleaning rule drops stubs and outlier walls of text, keeping both sources in the same length band so the comparison is fair:

```python
MIN_CHARS = 200          # drop stub postings
MAX_CHARS = 8000         # drop outlier walls of text (keeps both sources comparable)
N_TRAIN = 12_000         # ~19 MB of text - well above the brief's 1-10 MB floor

def clean(series):
    s = series.dropna().astype(str).str.strip()
    return s[s.str.len().between(MIN_CHARS, MAX_CHARS)]
```

The splits are drawn from a seeded shuffle so train, val, and test are disjoint Djinni documents, and the out-of-domain set is a deduplicated LinkedIn sample:

```python
rng = random.Random(SEED)
dj_texts = clean(dj["Long Description"]).tolist()
rng.shuffle(dj_texts)
train = dj_texts[:N_TRAIN]
val   = dj_texts[N_TRAIN:N_TRAIN + N_VAL]
test  = dj_texts[N_TRAIN + N_VAL:need]

lk_texts = clean(lk["description"]).drop_duplicates().tolist()
rng.shuffle(lk_texts)
ood = lk_texts[:N_OOD]
```

Output: `data/processed/mp1/{train,val,test,ood_test}.jsonl` at 12,000 / 1,000 / 1,000 / 1,000 documents. The 12k train documents come to about 22.7 MB of text (roughly 5.4 million SmolLM2-360M tokens), comfortably above the brief's 1 to 10 MB floor, which is a deliberate choice: more domain text gives the adaptation more to work with, and the 4090 can train on it in a reasonable time. (The `# ~19 MB` comment in the script above was an early pre-tokenisation estimate; the on-disk number is what the run actually consumed.)

Rationale for the LinkedIn deduplication: roughly 16k LinkedIn rows are reposts that share identical description text, so without `drop_duplicates` the out-of-domain test set would be skewed by copies.

## 5. Step 3 - Continued-pretraining setup

Continued pretraining is next-token prediction on raw domain text. The data has to be tokenized and packed into fixed-length blocks. `src/train.py` does this with a tokenize-then-group pass that marks each document boundary with the EOS token and packs the stream into blocks of 1024 tokens:

```python
def tokenize_and_chunk(jsonl_path, tokenizer, block_size):
    """Tokenize raw text, mark doc boundaries with EOS, pack into fixed blocks."""
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    def tok(batch):
        out = tokenizer(batch["text"])
        for ids in out["input_ids"]:
            ids.append(tokenizer.eos_token_id)   # document boundary
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

    def group(batch):
        ids = list(itertools.chain.from_iterable(batch["input_ids"]))
        n = (len(ids) // block_size) * block_size      # drop the ragged tail
        blocks = [ids[i:i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": blocks,
                "attention_mask": [[1] * block_size for _ in blocks]}

    return ds.map(group, batched=True)
```

Two reasons for this design. First, appending EOS between documents tells the model where one posting ends and the next begins, so packed blocks do not blur two unrelated postings into one training example without a signal. Second, fixed-size blocks keep every batch the same shape, which is the efficient way to feed a causal language model and avoids padding waste.

The base model is loaded without the instruct head, because this is continued pretraining and not instruction tuning:

```python
model: HuggingFaceTB/SmolLM2-360M     # base model (NOT -Instruct: this is continued pretraining)
seed: 42
block_size: 1024
```

The tokenizer has no pad token by default, so it is aliased to EOS; this is the standard pattern for this model family and is harmless because padding positions are masked out of the loss.

## 6. Step 4 - Training: the full-versus-LoRA experiment

The experiment is a single knob, `--mode full` or `--mode lora`, with everything else held constant in `configs/train.yaml`. The constant block and the mode-specific block are kept separate so only the intended variable changes:

```yaml
training:                  # held constant across both modes
  epochs: 3
  per_device_batch_size: 4
  grad_accum: 4            # effective batch size = 16
  warmup_ratio: 0.03
  weight_decay: 0.01
  bf16: true
  eval_strategy: steps
  eval_steps: 50

modes:                     # the experiment - only this section differs
  full:
    learning_rate: 5.0e-5  # standard range for full continued pretraining
  lora:
    learning_rate: 2.0e-4  # LoRA adapters need a higher LR than full FT
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

In LoRA mode the base weights are frozen and small low-rank adapters are attached to the attention and feed-forward projections:

```python
if args.mode == "lora":
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=mcfg["lora_r"], lora_alpha=mcfg["lora_alpha"],
        lora_dropout=mcfg["lora_dropout"],
        target_modules=mcfg["lora_target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
```

Training itself uses the HuggingFace `Trainer` with the standard causal-LM data collator (no masked language modelling), and writes a loss-curve log plus a summary on completion:

```python
trainer = Trainer(
    model=model, args=targs,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    processing_class=tokenizer,
)
trainer.train()
...
final = trainer.evaluate()
ppl = math.exp(final["eval_loss"])
(out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
(out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
```

Each run produces the expected wall-clock and trainable-parameter footprint on the 4090. Full fine-tuning updates all 361.8M model parameters and takes about 42 minutes; LoRA updates only 8.7M parameters (2.3 percent of the model) and takes about 13 minutes, roughly a third of the time.

Decisions and their rationale:

- Effective batch size of 16 is reached through micro-batch 4 and gradient accumulation 4. This was forced by a memory limit (see Section 9), and keeping the product fixed means the optimization behaviour does not change.
- The learning rates differ by design. Full fine-tuning uses 5e-5 and LoRA uses 2e-4, each its conventional value. They cannot be held equal without handicapping one method, so the comparison carries a learning-rate confound, which is reported honestly rather than hidden.
- bf16 mixed precision is used because the 4090 supports it and it halves activation memory.
- `save_strategy: no` during training plus an explicit `save_model` at the end keeps disk usage down (no mid-run checkpoints) while still producing the final artifact.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set at import time to reduce allocator fragmentation on long runs.

## 7. Step 5 - Evaluation

`src/evaluate_mp1.py` measures perplexity for three model states (base, full, lora) on two test sets (in-domain Djinni, out-of-domain LinkedIn) and records greedy generations for a fixed set of prompts.

Perplexity is computed as the exponential of the token-weighted mean loss over fixed 1024-token blocks, batched for speed and under `no_grad`:

```python
@torch.no_grad()
def perplexity(model, blocks):
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(blocks), 8):
        batch = torch.tensor(blocks[i:i + 8], device=DEVICE)
        loss = model(batch, labels=batch).loss
        n = batch.shape[0] * (batch.shape[1] - 1)
        tot_loss += loss.item() * n
        tot_tok += n
    return math.exp(tot_loss / tot_tok)
```

A detail that recurs across all three scripts: the base model id is never hardcoded. It is read from the LoRA adapter's config, so the evaluation always matches whatever base that run was trained on, and the same script works for the 135M and the 360M runs without edits:

```python
def base_model_id(run):
    """Base model id, read from the run's LoRA adapter config - never hardcoded."""
    cfg = json.loads((OUTPUTS / run / "lora" / "adapter_config.json").read_text())
    return cfg["base_model_name_or_path"]
```

Results are written to `outputs/<run>/eval.json`. The generations use greedy decoding here so the before/after comparison is deterministic and reproducible.

## 8. Step 6 - Interactive generation and deployment

`src/generate_mp1.py` is the deployment surface: a small CLI that loads any of the three model states for a run and completes prompts, either as a one-shot call, a side-by-side comparison of base/full/lora, or an interactive REPL.

Generation defaults to sampling with a repetition penalty, with a greedy flag for deterministic output:

```python
def generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8,
             top_p=0.95, greedy=False, seed=42):
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.3,
                  pad_token_id=tokenizer.eos_token_id)
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    out = model.generate(**enc, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

The repetition penalty of 1.3 is there because small continued-pretrained models loop easily; it curbs the degenerate "and the and the" failure mode without forcing greedy decoding.

## 9. Results and findings

Three-way test perplexity (lower is better), for both model sizes:

| variant     | 135M in-domain | 135M OOD | 360M in-domain | 360M OOD |
|-------------|----------------|----------|----------------|----------|
| base        | 20.43          | 23.97    | 16.37          | 17.80    |
| full FT     | 16.24          | 23.09    | 13.12          | 17.42    |
| LoRA r=16   | 13.64          | 23.94    | 11.38          | 18.27    |

What the numbers say:

1. Scale helps everywhere. The 360M base beats the 135M base by about 20 percent in-domain and about 26 percent out-of-domain; the larger out-of-domain gain means scale buys generalisation.
2. Adaptation helps in-domain and does almost nothing out-of-domain, at both sizes. In-domain improves about 30 to 33 percent with LoRA; out-of-domain is flat to slightly worse (the 360M LoRA at 18.27 is mildly above its base 17.80, a small overfit).
3. The 135M LoRA (13.64) beats the 360M base (16.37) in-domain: adapting a small model beats merely scaling up, in-domain.
4. The 360M base (17.80) beats every 135M variant out-of-domain: scale gives a robustness that adaptation alone cannot.

Best overall is the 360M LoRA at 11.38 in-domain, which is the checkpoint carried forward to Mini-Assignment 2. LoRA's edge over full fine-tuning is partly the learning-rate confound noted above, and the pattern is consistent at both sizes.

## 10. Key engineering decisions and difficulties

These are the process-level points, captured in the notebook's Section 5.7:

- LoRA hit CUDA out-of-memory at micro-batch 16 on this transformers and peft stack. The fix was micro-batch 4 with gradient accumulation 4, which keeps the effective batch size at 16 while fitting the card.
- The full-versus-LoRA comparison cannot hold the learning rate equal; each method keeps its conventional value, so the result carries a learning-rate confound. Reported, not hidden.
- transformers 5.9 removed a `TrainingArguments` option the first training script relied on (`overwrite_output_dir`), which had to be dropped.
- The project began on SmolLM2-135M. Its generations were weak and repetitive, which led to redoing the whole experiment at 360M. The 135M run is kept as a comparison baseline. What we would do differently: fix the model size deliberately at the start.
- Running two model sizes caused output files to collide, which forced the per-run `outputs/<run>/` scheme (each run holds `full/`, `lora/`, `eval.json`). The scripts became run-aware through a `--run` argument and a `load_model(name, run)` helper.
- What surprised us: how little the in-domain adaptation transferred out of domain. This is the headline finding and it is honest about the limits of domain adaptation.
- On a separate machine (an RTX 5060 Ti on Windows) the attention-backend fallback inflated memory use, because the FlashAttention kernel is not shipped in the Windows wheels; the WSL2 Linux wheels do ship it. This is why the work settled on WSL2.

The choice of the SmolLM2 family itself was deliberate: it is an open, modern small-model family, and crucially all sizes share one tokenizer, which made the 135M-to-360M comparison friction-free (no tokeniser remap, the corpus and the evaluation code are identical across sizes). 360M was chosen as the main model because it fits the 4090 for full fine-tuning and clearly beats 135M at roughly double the compute.

## 11. Deliverables

The frozen MA1 notebook is `src/atlm_mp1_v4.ipynb` (113 cells, six CRISP-DM phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment). The released bundle is `mp1/delivery/`: `atlm_ma1_groupc.ipynb`, `atlm_ma1_report_groupc.pdf`, `README.md`, a pinned `requirements.txt`, and `src/generate_mp1.py`. Trained artifacts live under `outputs/mp1-360m/{full,lora}/` with `outputs/mp1-360m/eval.json`, and the 135M baseline under `outputs/mp1-135m/`. The 360M LoRA adapter is the starting point for Mini-Assignment 2.
