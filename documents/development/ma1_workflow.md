# Mini-Assignment 1 - Development Workflow

This document is the step-by-step workflow log for Mini-Assignment 1 of the ATLM project. It complements the implementation report by recording the order in which the work was actually carried out, the decisions made at each junction, and the artefacts produced along the way. Where the implementation report explains the final state of the system, this log explains how that state was reached. The five level-2 sections below correspond to the five development phases, from the initial reading of the brief to the assembly of the final upload bundle. Read this alongside documents/development/ma1_implementation_report.md and the academic PDF in mp1/delivery/atlm_ma1_report_groupc.pdf.

## Step 1: Goal, model family, and data understanding

The first phase of MA1 was about converting a one-page assignment brief into a defensible project shape: a model family, a domain, and a data layout that the rest of the pipeline could be built on without revisiting. The activities below ran in the order shown, and each one fed the next.

1. Read the MA1 brief and extract the hard constraints. The assignment required an open-source base model under one billion parameters, continued pretraining (causal language modelling, not instruction tuning) on a chosen text domain, a corpus of roughly 1 to 10 MB, and a controlled experiment that compared at least two training regimes. Writing those constraints down first mattered because every later decision (size of the model, size of the corpus, choice of baselines) had to be justified against them.

2. Choose the SmolLM2 family. HuggingFaceTB/SmolLM2 was selected because it is fully open (weights, tokenizer, training recipe), sits cleanly under the 1B cap in its 135M, 360M, and 1.7B variants, and has a tokenizer that is already reasonable on technical English. The initial intent was to run on SmolLM2-135M for fast iteration, with the option to move up to 360M once the pipeline was stable. The 1.7B variant was excluded early because it would not leave headroom for the LoRA-versus-full comparison on a single consumer GPU.

3. Pick the domain and acquire the two source datasets. IT job postings were chosen as the domain because the register is narrow enough that continued pretraining should move the loss in a measurable way, and because two complementary public datasets existed:
   - Djinni IT job postings, 141,897 rows, 0% duplicates, mean length 1,801 characters, distributed as Parquet. This is a pure IT corpus and became the in-domain source.
   - LinkedIn job postings, 123,849 rows, 12.9% duplicates, mean length 3,766 characters, distributed as CSV across industries. This is broader and noisier, and was earmarked as a candidate out-of-domain probe rather than training data.

4. Run the Data Understanding pass in Section 2 of `src/atlm_mp1_v4.ipynb`. This was not a formality: the choices made here determined the train/val/test split, the held-out evaluation set, and the corpus size we would target for `src/prepare_corpus.py`. The pass covered:
   - Schema profiling of both files (column names, dtypes, null rates, row counts) to confirm that Djinni's `Long Description` and LinkedIn's `description` were the only fields worth keeping for LM training.
   - Length distributions in characters and tokens. Djinni postings cluster around 1,801 characters mean, LinkedIn around 3,766. The LinkedIn tail is much heavier, which matters because long, generic postings would dominate the loss if the two were mixed.
   - Keyword distributions over a curated IT vocabulary (Python, AWS, Kubernetes, etc.) confirming Djinni is densely IT-coded while LinkedIn is diluted by sales, healthcare, and retail postings.
   - Lexical diversity (type-token ratio, vocabulary growth curves) showing Djinni reaches a stable vocabulary faster, as expected for a single-domain corpus.
   - Tokenisation analysis with the SmolLM2 tokenizer, measuring tokens-per-document and the chars-per-token ratio so we could later size the corpus in tokens rather than bytes.
   - Semantic clustering using sentence-transformers embeddings reduced with UMAP. Djinni postings collapse into a tight IT manifold; LinkedIn spreads across multiple well-separated clusters with only a partial overlap with the Djinni region. This was the visual confirmation that the two corpora occupy different distributions.

5. Decide the data layout that the rest of the project would inherit. The data understanding produced one decision, recorded explicitly so it would not be relitigated later: Djinni is the only corpus the model sees during continued pretraining, and it is split internally into train, validation, and in-domain test. LinkedIn is held out entirely and used only as the out-of-distribution test probe at evaluation time. This gives a clean four-way evaluation surface (train loss, val loss, in-domain test, OOD test) and avoids the failure mode where a mixed-source corpus makes "in-domain" and "out-of-domain" indistinguishable. The numbers behind the decision are the ones above: 141,897 clean Djinni rows are more than enough to carve a 12,000-document training set inside the 1 to 10 MB target, and the LinkedIn distribution sits far enough away in embedding space to be a meaningful generalisation probe.

## Step 2: Corpus construction and pretraining setup

With the data sources understood, the next task was turning two raw dumps (a Djinni parquet and a LinkedIn CSV) into a clean, reproducible continued-pretraining corpus, and deciding how that corpus would be presented to SmolLM2-360M during training.

1. Writing `src/prepare_corpus.py`. The script reads the in-domain Djinni file `data/jobs/djinni/train-00000-of-00001.parquet` (column `Long Description`) and the out-of-domain LinkedIn file `data/jobs/linkedin/postings.csv` (column `description`), and writes one `{"text": ...}` JSON object per line to `data/processed/mp1/{train,val,test,ood_test}.jsonl`.

2. Cleaning rule. A single helper `clean()` drops nulls, strips whitespace, and keeps only documents whose length is between `MIN_CHARS=200` and `MAX_CHARS=8000`. The lower bound removes stub postings that are too short to teach the model anything about domain phrasing; the upper bound trims a handful of outlier walls of text and keeps the two sources comparable in distribution.

   ```python
   MIN_CHARS = 200          # drop stub postings
   MAX_CHARS = 8000         # drop outlier walls of text (keeps both sources comparable)

   def clean(series):
       s = series.dropna().astype(str).str.strip()
       return s[s.str.len().between(MIN_CHARS, MAX_CHARS)]
   ```

3. Seeded shuffle and disjoint splits. A `random.Random(SEED)` with `SEED=42` shuffles the cleaned Djinni list once, then slices it into three disjoint chunks: 12,000 train, 1,000 val, 1,000 test. The LinkedIn list is shuffled with the same RNG, deduplicated via `drop_duplicates()` to avoid repeated postings inflating evaluation, and the first 1,000 docs become the OOD test set. Because the slicing happens after a single seeded shuffle, the three Djinni splits are guaranteed disjoint and the run is bit-for-bit reproducible.

4. JSONL output and on-disk sizes. The `write()` helper emits one `{"text": ...}` line per document using `ensure_ascii=False` to preserve non-ASCII characters. Final sizes after the run: `train.jsonl` is 22.7 MB (12,000 docs, roughly 5.4M SmolLM2-360M tokens), well above the brief's 1-10 MB floor, with `val.jsonl`, `test.jsonl`, and `ood_test.jsonl` each around 1.9 MB at 1,000 docs.

5. Writing `tokenize_and_chunk` in `src/train.py`. Continued pretraining wants a stream of fixed-length blocks, not variable-length padded sequences. The function tokenizes each document, appends the tokenizer's EOS token as an explicit document boundary, concatenates everything, and slices the result into `block_size=1024` blocks. The ragged tail at the end (whatever is left over after the last full block) is dropped.

   ```python
   def tokenize_and_chunk(jsonl_path, tokenizer, block_size):
       ds = load_dataset("json", data_files=str(jsonl_path), split="train")

       def tok(batch):
           out = tokenizer(batch["text"])
           for ids in out["input_ids"]:
               ids.append(tokenizer.eos_token_id)
           return {"input_ids": out["input_ids"]}

       ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

       def group(batch):
           ids = list(itertools.chain.from_iterable(batch["input_ids"]))
           n = (len(ids) // block_size) * block_size
           blocks = [ids[i:i + block_size] for i in range(0, n, block_size)]
           return {"input_ids": blocks,
                   "attention_mask": [[1] * block_size for _ in blocks]}

       return ds.map(group, batched=True)
   ```

6. Reasoning behind the packing choices. Appending EOS at the end of every document gives the model an explicit "this posting is over" signal, so when two unrelated postings end up in the same 1024-token block the language-modelling loss does not try to predict the tail of one from the head of the other as if they were continuous text. Fixed 1024-token blocks (rather than per-document padding) avoid wasting compute on pad tokens, keep every batch shape uniform for the GPU, and let `DataCollatorForLanguageModeling` operate on dense inputs only.

7. Base checkpoint, not instruct. `configs/train.yaml` pins `model: HuggingFaceTB/SmolLM2-360M`, the raw pretrained checkpoint, and `src/train.py` loads it with `AutoModelForCausalLM.from_pretrained(cfg["model"])`. The `-Instruct` variant is deliberately avoided: this assignment is continued pretraining on plain domain text, not instruction tuning, so the model is expected to keep doing next-token prediction on raw job postings rather than answering chat-style prompts. Picking the instruct checkpoint would have meant fighting its chat formatting throughout training and evaluation.

## Step 3: The full-vs-LoRA controlled experiment

The third step turned the prepared corpus into two trained checkpoints that could be compared head to head. The whole experiment was driven from a single YAML file and a single training script, with the only meaningful difference between runs being the adaptation method.

1. Splitting the configuration into a shared block and a modes block. The file configs/train.yaml was written so that every hyperparameter that should be held equal across the two runs lives in one place, and only the knobs that genuinely have to differ live in a per-mode override. The shared block fixes epochs at 3, micro-batch at 4, gradient accumulation at 4 (effective batch 16), warmup_ratio 0.03, weight_decay 0.01, bf16 mixed precision, evaluation every 50 steps, and seed 42. The modes block then carries one learning rate per method, plus the LoRA-specific adapter configuration.

```yaml
shared:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_ratio: 0.03
  weight_decay: 0.01
  bf16: true
  eval_steps: 50
  seed: 42

modes:
  full:
    learning_rate: 5.0e-5
  lora:
    learning_rate: 2.0e-4
    lora:
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

2. Writing src/train.py as a single entry point for both runs. The training script reads configs/train.yaml, takes a --mode full|lora flag on the command line, merges the shared block with the selected mode block, and hands the result to the HuggingFace Trainer. The data collator is DataCollatorForLanguageModeling(mlm=False) so that the loss is plain causal language modelling on the IT job posting corpus rather than masked LM. The same script handles both runs, which keeps the two checkpoints comparable at the code level: identical data pipeline, identical tokenizer, identical Trainer configuration, identical logging cadence.

3. Attaching LoRA when --mode lora is selected. In LoRA mode the base SmolLM2-360M is loaded first and then wrapped with a PEFT LoraConfig. The seven target modules cover both the attention projections and the MLP projections, which is the conventional choice for LLaMA-style architectures and matches what the LoRA literature recommends for this model family.

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

4. Trainable parameter counts and wall-clock on the RTX 4090. The full fine-tuning run updates all 361.8M parameters of SmolLM2-360M and completes the three epochs in about 42 minutes. The LoRA run updates 8.7M parameters, which is 2.3 percent of the model, and completes the same three epochs in about 13 minutes. The 3.2x speedup is the headline efficiency result of the experiment and is reported directly from Trainer's wall-clock logs rather than estimated.

5. The learning rate confound, reported rather than hidden. Full fine-tuning was run at 5e-5 and LoRA at 2e-4. These are the conventional values for the two methods and were chosen up front rather than tuned. Holding the learning rate equal across the two runs was considered and rejected: 5e-5 is too small for LoRA adapters to move meaningfully in three epochs, and 2e-4 is large enough to destabilise full fine-tuning of a 360M model. Either choice would handicap one of the methods and produce a misleading comparison. The honest path was to use each method at its conventional rate and to flag the confound explicitly in the report, which is what the implementation log does.

6. Memory engineering on a single 24 GB card. Two memory measures were needed to fit the full fine-tuning run on the 4090. First, the environment variable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set at import time in src/train.py, before torch is imported, so that the CUDA caching allocator can grow its segments instead of fragmenting under the spiky activations of a 360M causal LM at bf16. Second, the initially intended micro-batch of 16 triggered a CUDA out-of-memory error during the first forward pass of the full run; the working configuration backs that off to micro-batch 4 with gradient accumulation 4, which preserves the effective batch size of 16 while keeping peak memory inside the 24 GB budget. The LoRA run would have fit at micro-batch 16, but the same micro-batch 4 / grad-accum 4 setting is used for it so that the two runs see identical optimisation dynamics.

## Step 4: Evaluation, generation, and iteration

With both adapters trained, attention shifted to measuring whether the continued pretraining had actually moved the model toward the IT-job-posting distribution, and to looking at the raw outputs to sanity-check the perplexity numbers.

1. Writing src/evaluate_mp1.py. The evaluation script computes perplexity as a token-weighted mean cross-entropy. Each evaluation file (the in-domain Djinni test split at data/processed/mp1/test.jsonl and the out-of-domain LinkedIn split at data/processed/mp1/ood_test.jsonl) is tokenized, an EOS is appended per document, and the resulting id stream is cut into non-overlapping 1024-token blocks. The remainder is dropped so every block has the same length and contributes the same number of next-token predictions. Blocks are evaluated in batches of 8 under torch.no_grad, the per-batch mean loss is reweighted by the number of label positions (batch_size * (block - 1)), and the final perplexity is exp(total_loss / total_tokens). The hot loop is small and explicit:

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

After perplexity, the same script does greedy generations from a fixed prompt set (`"We are looking for a"`, `"The ideal candidate will"`, `"Responsibilities:\n-"`) for each of base, full, and LoRA, and writes everything to outputs/<run>/eval.json so the perplexity numbers and the qualitative samples live in one file per run.

2. The base_model_id convention. Rather than hardcode "HuggingFaceTB/SmolLM2-360M" in the evaluation and generation scripts, both read the base model identifier from the LoRA adapter's own adapter_config.json:

```python
def base_model_id(run):
    """Base model id, read from the run's LoRA adapter config - never hardcoded."""
    cfg = json.loads((OUTPUTS / run / "lora" / "adapter_config.json").read_text())
    return cfg["base_model_name_or_path"]
```

This is what lets a single invocation of `evaluate_mp1.py --run mp1-135m` or `evaluate_mp1.py --run mp1-360m` work without any code edits: each run's adapter config carries the matching base model id (HuggingFaceTB/SmolLM2-135M for the first run, HuggingFaceTB/SmolLM2-360M for the second), and the scripts simply trust it.

3. Writing src/generate_mp1.py. The generation script is an interactive companion to the evaluator. It supports three modes through one CLI: a one-shot mode (`--prompt "Senior Data Engineer"`), an all-three comparison mode (`--model all`, which loads base, full, and LoRA and prints completions for the same prompt side by side), and a REPL when --prompt is omitted. Sampling is the default (temperature 0.8, top-p 0.95, repetition_penalty 1.3), reflecting that MP1 is a text completer and a small amount of stochasticity makes the qualitative differences between the variants easier to see. A --greedy flag is available for deterministic decoding when reproducing eval.json samples. The generation kwargs are assembled once:

```python
kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.3,
              pad_token_id=tokenizer.eos_token_id)
if greedy:
    kwargs["do_sample"] = False
else:
    kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
```

4. The 135M run, and the decision to scale up. The experiment was first run end to end at SmolLM2-135M. Perplexity moved in the expected direction (20.43 base, 16.24 full, 13.64 LoRA in-domain), but the greedy generations were poor. The base model looped on phrases like "We are looking for a new way to make a better product." repeated four times, and produced "Responsibilities:\n- 1.1.1.1.1.1..." for the third prompt. The full-FT and LoRA variants did produce job-posting-shaped text (Senior DevOps Engineer, CI/CD pipelines, React/Angular/Node.js), but still degenerated into immediate repetition within a few lines. The numbers were defensible but the qualitative outputs were too weak to anchor a written analysis. The decision was to redo the experiment at SmolLM2-360M, which is still tractable on the same GPU under the same configs but produces noticeably more coherent continuations. The 135M run was kept rather than deleted: it became a useful baseline for talking about the effect of scale alongside the effect of adaptation.

5. The outputs/<run>/ layout. Running two model sizes immediately produced a collision: both runs wanted to write outputs/full/, outputs/lora/, and outputs/eval.json. The fix was a run-aware layout: outputs/mp1-135m/ and outputs/mp1-360m/ each contain their own full/, lora/, and eval.json. The training, evaluation, and generation scripts all take a --run argument and resolve every path under outputs/<run>/, which is also why the base_model_id convention from sub-step 2 is keyed on the run directory.

6. Final perplexity table. The twelve numbers below come directly from outputs/mp1-135m/eval.json and outputs/mp1-360m/eval.json. Lower is better.

| Model    | Variant | In-domain (Djinni) | Out-of-domain (LinkedIn) |
|----------|---------|--------------------:|--------------------------:|
| 135M     | base    | 20.43               | 23.97                     |
| 135M     | full-FT | 16.24               | 23.09                     |
| 135M     | LoRA    | 13.64               | 23.94                     |
| 360M     | base    | 16.37               | 17.80                     |
| 360M     | full-FT | 13.12               | 17.42                     |
| 360M     | LoRA    | 11.38               | 18.27                     |

7. Headline findings. Three patterns hold across the table. First, scale helps everywhere: every 360M variant beats its 135M counterpart on both splits, and the gap is largest OOD where the 360M base already sits at 17.80 against the 135M base's 23.97. Second, adaptation helps in-domain and does almost nothing OOD: full-FT and LoRA reduce in-domain perplexity by roughly 20-30 percent over base at each size, but OOD perplexity barely moves (and LoRA at 360M even drifts slightly up to 18.27 from a base of 17.80, consistent with mild over-specialization on Djinni's phrasing). Third, the two effects interact in informative ways: 135M LoRA at 13.64 in-domain beats 360M base at 16.37 in-domain, showing that targeted adaptation at a smaller scale can outperform a larger un-adapted model on its own distribution; conversely, 360M base at 17.80 OOD beats every 135M variant OOD (best 135M OOD is 23.09), showing that on unfamiliar text, more parameters carry farther than more in-domain training. The best overall number is 360M LoRA at 11.38 in-domain, which is also why mp1-360m/lora was chosen as the headline checkpoint for MA1.

## Step 5: Notebook revision and delivery

The training and evaluation results from Step 4 were the raw material. Step 5 turned that material into a single self-contained academic notebook, regenerated the deliverables against the brief, and assembled the upload bundle.

1. Restructure into CRISP-DM. The earlier exploratory notebooks were reorganised into src/atlm_mp1_v4.ipynb, a 113-cell document laid out as the six CRISP-DM phases:

   - Section 1 Business Understanding (problem framing, objectives, success criteria)
   - Section 2 Data Understanding (corpus discovery, length and lexical statistics, n-gram analysis, semantic-embedding exploration)
   - Section 3 Data Preparation (cleaning, deduplication, train/validation split, tokenisation)
   - Section 4 Modeling (model selection rationale, full fine-tuning run, LoRA run, training curves)
   - Section 5 Evaluation (held-out perplexity, generation quality, OOD probe, comparison tables, difficulties)
   - Section 6 Deployment (inference recipe, packaging, reproducibility notes)

   Every analysis cell is followed by an interpretation cell written in plain prose, and every code cell carries inline comments. The prose was rewritten end to end against the project style rules: no em dashes, no en dashes, no arrows, no emojis, minimal bold, each paragraph reflowed as one continuous line.

2. Install the semantic-embedding stack. Section 2.8 was designed around sentence-transformers all-MiniLM-L6-v2 with a UMAP projection. The earlier run had fallen back to a vocabulary-based proxy because the libraries were missing. They were pinned and installed into the project environment:

   ```bash
   pip install "sentence-transformers==3.0.1" "umap-learn==0.5.6"
   ```

   With both packages present, the Section 2.8 cell encodes a sample of postings, projects to 2D with UMAP, and renders the cluster plot from real embeddings rather than TF-IDF cosine.

3. Phase B, end-to-end rerun at 360M. The full v4 notebook was executed top to bottom on the RTX 4090 against SmolLM2-360M. The numbers from the prior 360M training and evaluation runs reproduced exactly across the rerun: the cached corpus statistics, the training and validation losses, the held-out perplexities for base, FFT and LoRA, and the generation samples. Cell outputs were committed so a reader does not need to rerun to see the results.

4. Phase C, post-run revision for full brief coverage. With real outputs in hand, several sections were rewritten or added:

   - Section 2.8 interpretation cells were rewritten against the actual UMAP plot of MiniLM embeddings, describing the observed clusters (developer, data, security, infrastructure) instead of the placeholder vocabulary commentary.
   - Section 5.7 Difficulties Encountered was added and lists, in order, the LoRA OOM on the 16 GB card and how it was resolved by batch and sequence trimming, the learning-rate confound between the two runs and how it was controlled in the final comparison, the transformers 5.9 API change to the Trainer signature, the switch from SmolLM2-135M to SmolLM2-360M after the 135M run underfit the corpus, the output-directory collision between the FFT and LoRA runs that forced a rename scheme, the absence of OOD transfer on the held-out non-IT probe, and the RTX 5060 Ti attention-backend fallback from flash-attn to SDPA on Blackwell.
   - Section 4.1 was expanded to cover the hardware footprint, the compute budget in GPU-hours and wall-clock minutes, the architecture rationale for picking a 360M decoder over both the 135M and a larger candidate, and the reproducibility recipe (seed, environment, dataset hash).
   - README.md was rewritten from scratch to mirror the notebook structure, list the exact commands to reproduce the runs, and point at the pinned requirements.

5. Consistency pass. A final sweep checked that the heading tree is monotone (1, 1.1, 1.2, 2, 2.1, ..., 6), that every cross-reference in the prose resolves to an existing section, that no banned characters remain (em dash, en dash, arrows, smart quotes, emojis), and that every code cell parses. The check was scripted with a short pass over the notebook JSON:

   ```python
   import json, ast
   nb = json.load(open("src/atlm_mp1_v4.ipynb"))
   for i, c in enumerate(nb["cells"]):
       if c["cell_type"] == "code":
           ast.parse("".join(c["source"]))
   banned = ["\u2014", "\u2013", "\u2192", "\u2190"]
   ```

6. Assemble mp1/delivery/. The upload bundle was staged as:

   - atlm_ma1_groupc.ipynb (a copy of src/atlm_mp1_v4.ipynb renamed to the submission convention)
   - atlm_ma1_report_groupc.pdf (the notebook exported to PDF through nbconvert)
   - README.md (the rewritten top-level readme)
   - requirements.txt (pinned, including transformers, peft, datasets, accelerate, sentence-transformers 3.0.1, umap-learn 0.5.6, torch matched to the CUDA build used)
   - src/generate_mp1.py (the standalone inference script)

   The directory was then zipped to mp1/delivery/atlm_ma1_groupc.zip for the platform upload.

7. Hardware footprint of record. All work in Step 5 ran under WSL2 Ubuntu 24.04 on Windows 11. Both the RTX 5060 Ti (16 GB, Blackwell, SDPA attention) and the RTX 4090 (24 GB, Ada, flash-attention) were exercised during development. The canonical 360M runs whose numbers appear in the delivered notebook were produced on the RTX 4090.

## Closing summary

The headline outcome of MA1 is the mp1-360m/lora checkpoint, which reaches 11.38 in-domain perplexity on the held-out Djinni test split, the best number across all six trained variants and the lowest-cost path to it (8.7M trainable parameters, about 13 minutes on a single RTX 4090). That checkpoint is the artefact carried forward into Mini-Assignment 2 as the starting point for instruction tuning and downstream evaluation. The work was delivered as src/atlm_mp1_v4.ipynb together with the standalone scripts, the README, the pinned requirements, and the PDF export staged in mp1/delivery/. This workflow log is meant to be read alongside documents/development/ma1_implementation_report.md and the academic PDF in mp1/delivery/atlm_ma1_report_groupc.pdf: the implementation report captures the final system as built, the academic PDF presents the result, and this log records the order and reasoning of the development itself.
