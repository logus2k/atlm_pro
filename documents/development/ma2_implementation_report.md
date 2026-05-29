# Mini-Assignment 2 - Implementation Report

This document records how Mini-Assignment 2 (alignment) is being built: the development steps in order, what each step implements, the most relevant code, and the reasoning behind the options chosen. It is a development log, not the academic report. The academic report is the notebook `src/ma2/atlm_ma2_v2.ipynb`.

Mini-Assignment 2 is in progress. Three stages are implemented and one of them has been run end to end; the remaining stages are designed, with their decisions taken, but not yet coded. This document is explicit about which is which.

## 1. Goal and pipeline

The brief asks us to take the domain-adapted model from Mini-Assignment 1 and apply an alignment technique on top of it, walking the full post-training pipeline and reasoning about what each stage changes. The report cap is 5 pages.

The pipeline has four stages, the last two belonging to this assignment:

1. Pretraining: SmolLM2-360M, trained by its authors on general text.
2. Continued pretraining (Mini-Assignment 1): adapted to the job-postings domain by next-token training on raw IT descriptions. Produces a domain-fluent text completer.
3. Supervised fine-tuning (this assignment): teaches the completer to follow recruiter instructions, using (query, structured posting) pairs.
4. Preference alignment (this assignment): Direct Preference Optimization (DPO) with reinforcement learning from AI feedback (RLAIF), using a stronger model as judge.

The chosen alignment technique is DPO via RLAIF. DPO is the recommended default for a single-GPU budget (no separate reward model, no PPO instability), and RLAIF fits because a capable judge model (Gemma-4) is already available in the project as the `atlm_teacher` agent, so preferences can be generated rather than hand-annotated or pulled from a generic dataset.

## 2. Tooling and environment

The alignment stack is HuggingFace TRL 1.5.1, transformers 5.9.0, peft 0.19.1, and datasets, in the project virtual environment `.venv_atlm_pro` (Python 3.12), on a single RTX 4090 (24 GB). TRL was pinned at 1.5.1 with no version churn on transformers, torch, or peft. The seed is 42 throughout.

A note on the TRL API at this version, since it changed from older tutorials: `SFTConfig` takes `max_length` (not `max_seq_length`) and `eval_strategy` (not `evaluation_strategy`), and `SFTTrainer` takes `processing_class` and `peft_config`. The notebook code uses the current names.

## 3. Step 0 - Supervised fine-tuning data preparation

The SFT data did not exist as a labelled set; it was generated from the same raw postings used in Mini-Assignment 1, by an ETL agent. The agent, `atlm_teacher`, is a Gemma-4 model served by a local `agent_server` (Docker, port 7701). Given a raw posting it returns three short recruiter queries plus a clean, structured job description in Markdown, or a skip marker for unusable input.

The output contract the agent follows: a `<QUERIES>` block of three numbered recruiter queries and a `<JOB_DESCRIPTION>` block (Markdown with the headings `## Summary`, `## Required Skills`, `## Responsibilities`, `## Requirements`), or `<SKIP>reason</SKIP>`. The parser tolerates a missing closing `</JOB_DESCRIPTION>` tag, because the agent stops on that token.

Before the bulk run, `src/validate_teacher.py` checks the agent on a small, deliberately diverse sample (one posting per tech keyword, plus very short and very long stress cases, plus cross-industry LinkedIn postings). It parses each output, checks format compliance, flags leaks (email, url, the word "apply"), and measures latency:

```python
def evaluate(parsed):
    if parsed["is_skip"]:
        return dict(verdict="SKIP", issues=[], leaks=[])
    issues, jd = [], parsed["jd"]
    if len(parsed["queries"]) != 3:
        issues.append(f"{len(parsed['queries'])} queries (expected 3)")
    if not re.search(r"^#\s+\S", jd, re.M):
        issues.append("no H1 title")
    for s in SECTIONS:
        if s not in jd:
            issues.append(f"missing '{s}'")
    leaks = []
    if re.search(r"[\w.]+@[\w.]+\.\w+", jd): leaks.append("email")
    if re.search(r"https?://", jd):          leaks.append("url")
    if re.search(r"\bapply\b", jd, re.I):    leaks.append("'apply'")
    return dict(verdict="OK" if not issues else "FAIL", issues=issues, leaks=leaks)
```

The bulk ETL, `src/run_etl.py`, interleaves both sources with a seeded shuffle, calls the agent with a worker pool, and writes one JSON record per successful conversion. It is append-only and resumable: the output file itself is the progress marker, so the job can be stopped and restarted, and it is bounded per run by `--count` and/or `--minutes`:

```python
def load_done():
    """IDs already converted or skipped - used to resume."""
    done = set()
    for path in (CONVERTED, SKIPPED):
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    done.add(json.loads(line)["id"])
    return done
```

Each converted record is written as a self-describing line, carrying the source so Djinni and LinkedIn stay separable downstream:

```python
conv_f.write(json.dumps(
    {"id": w["id"], "source": w["source"], "source_id": w["source_id"],
     "queries": parsed["queries"], "job_description": parsed["jd"],
     "converted_at": now}, ensure_ascii=False) + "\n")
```

The result used for SFT is `data/processed/converted.jsonl`: 2,507 records, each with three queries and one structured job description, which fan out to roughly 7,500 (query, posting) training pairs.

Rationale for generating the data this way: the assignment needs instruction pairs in the job-postings domain. A generic instruction dataset would not match the domain; manual annotation of thousands of pairs is infeasible. Using a stronger model to convert raw postings into clean, consistently structured targets is exactly the RLAIF idea applied at the data-preparation stage, and it produces targets whose Markdown structure the SFT model can learn to imitate.

## 4. Step 1 - Merge the Mini-Assignment 1 LoRA into the base

Mini-Assignment 1 produced both a full fine-tune and a LoRA adapter; the LoRA variant was selected as the best in-domain fit and carried forward. The first MA2 decision was whether to stack a new adapter on the existing LoRA or merge the LoRA into the base first.

Decision: merge. Merging produces one consolidated base, so each later stage (SFT, then DPO) trains a fresh LoRA on top of a plain model with no stacked-adapter bookkeeping, no question of which adapter is active, and no compounding of adapter-on-adapter numerical effects. The cost is a one-off 694 MB checkpoint on disk, which is acceptable.

The merge is implemented twice for two purposes. `src/ma2/merge_ma1_lora.py` is a standalone CPU-only script that hides the GPU from its own process, reads the base id from the adapter config, merges, and writes a provenance note:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide the GPU from this process
...
adapter_cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
BASE = adapter_cfg["base_model_name_or_path"]
...
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, str(LORA_DIR), torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model.save_pretrained(OUT_DIR)
```

The notebook also re-runs the merge inline in Section 2.1 for end-to-end reproducibility. One detail worth recording: the notebook merge cell originally set `os.environ["CUDA_VISIBLE_DEVICES"] = ""` like the standalone script, but that environment variable hides the GPU for the entire kernel session, which would have forced the later SFT cells onto the CPU. The fix was to remove that line from the notebook cell; the merge still runs on CPU because the weights load to CPU by default and no tensor is moved to the GPU, so the GPU stays free for training. The standalone script keeps the variable because it is a separate, short-lived process.

The merged model is saved to `outputs/mp1-360m/merged/` and loaded in Section 2.2; downstream stages do not need to know about the MA1 LoRA at all.

## 5. Step 2 - Supervised fine-tuning

This stage turns the domain-fluent completer into an instruction follower. It is implemented in Section 3 of the notebook and has been run.

### 5.1 Prompt template (Section 3.1)

The template is Alpaca-inspired: a one-sentence system preamble and two delimited fields.

```
You are a recruitment assistant. Given a brief recruiter request, write a complete structured job posting in Markdown.

### Request
{query}

### Posting
{jd}
```

Three options were considered: this one, a leaner key-value template, and a ChatML chat-template setup. This one was chosen for three reasons. First, explicit task framing: SmolLM2-360M is small and has never been instruction-tuned, so a one-sentence preamble gives it an unambiguous anchor for the task. Second, no clash with content: the structured postings already use `##` headings, so the template uses `###` for its separators and the model can tell a template marker from a response heading. Third, no special-token machinery: plain-text separators tokenise into existing vocabulary, so no new embeddings and no chat template need to be added to the tokenizer. The same template is used at inference, with the prompt ending at `### Posting\n`.

### 5.2 Training data (Section 3.2)

The 2,507 records are loaded, the three queries per record are fanned out into independent examples, and the split is done at the record level so no job description leaks across train and validation:

```python
random.Random(SEED).shuffle(records)
n_val = max(1, len(records) // 10)
val_records   = records[:n_val]
train_records = records[n_val:]

def fan_out(rs):
    out = []
    for r in rs:
        for q in r["queries"]:
            out.append({"text": format_example(q, r["job_description"])})
    return out
```

Splitting at the record level rather than the query level is the important choice here: if the split were per query, two phrasings of the same posting could land on opposite sides, and the validation loss would be measured partly on a target the model had already seen. The fan-out itself is a cheap augmentation, pairing one target with three phrasings so the model learns to be robust to how a request is worded.

### 5.3 Training configuration (Section 3.3)

A fresh LoRA is trained on the merged base, with the same shape that worked in Mini-Assignment 1 and the same effective batch size:

```python
SFT_CFG = {
    "epochs": 2,
    "per_device_batch": 4,
    "grad_accum": 4,                   # effective batch = 16
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
    "seed": SEED,
}
```

Two epochs is deliberately short: this is a light instruction pass, not a second round of domain adaptation, and over-training here would erode the domain knowledge from Mini-Assignment 1. The maximum sequence length of 1024 fits every pair without truncation. The loss is computed over the whole formatted sequence rather than only the response span; for a small model this is a minor inefficiency, not a correctness problem, and it keeps the configuration simple.

### 5.4 Running SFT (Section 3.4)

`run_sft()` wraps the merged model in the LoRA, configures the TRL `SFTTrainer`, trains, and writes the adapter plus a summary:

```python
args = SFTConfig(
    output_dir=str(SFT_OUT),
    num_train_epochs=SFT_CFG["epochs"],
    per_device_train_batch_size=SFT_CFG["per_device_batch"],
    gradient_accumulation_steps=SFT_CFG["grad_accum"],
    learning_rate=SFT_CFG["learning_rate"],
    max_length=SFT_CFG["max_seq_length"],
    bf16=True,
    eval_strategy="steps", eval_steps=100, logging_steps=20,
    save_strategy="no", seed=SFT_CFG["seed"], report_to=[],
    dataset_text_field="text",
)
trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=sft_train, eval_dataset=sft_val,
    peft_config=peft_cfg, processing_class=tokenizer,
)
```

The TRL trainer appends the EOS token to every formatted example, so the model learns when to stop. The tokenizer has no pad token, so it falls back to EOS; padding positions are masked out of the loss, so this is harmless.

### 5.5 Result and sanity check (Section 3.5)

The run completed in 18.8 minutes on the 4090. It trained 8.68M LoRA parameters (2.3 percent of the 370.5M total), over 6,771 train and 750 validation examples, reaching a final validation loss of 1.4955, a perplexity of 4.46. The 750 validation examples equal 250 held-out records times three queries, confirming the record-level split held.

The sanity check in Section 3.5 generates from the merged MA1 model and the SFT model side by side, with greedy decoding, on a few recruiter prompts. The expected and observed change is that the SFT model treats the request as an instruction and emits a structured posting with the right headings, where the pre-SFT model continues the text as free completion. This was confirmed qualitatively before freezing the run.

## 6. Step 3 - Three-way evaluation prompt set (Section 6.1)

The evaluation prompt set was frozen early, before any further training, so the three model states are measured on identical inputs and there is no opportunity to pick favourable prompts after the fact. The set is 20 recruiter queries, split into two ten-prompt sub-sets, persisted to `data/processed/ma2/eval_prompts.jsonl`:

```python
EVAL_PROMPTS = [
    {"id": "ind-01", "subset": "in_distribution",
     "query": "We need a backend engineer to build and maintain Python services on AWS, ..."},
    ...
    {"id": "ood-01", "subset": "out_of_distribution",
     "query": "We are hiring a junior developer straight out of a coding bootcamp; ..."},
    ...
]
```

The two sub-sets test different things. The ten in-distribution prompts are written in the style of the training data but are not part of it, so they are the fair measure of in-domain quality. The ten out-of-distribution prompts are fresh and probe generalisation: a junior bootcamp hire with no fixed stack, a principal staff engineer, niche stacks (Elixir and OTP, low-latency C++, Rust and WebAssembly), a freelance documentation writer, a soft-skills-heavy role, and a remote-async role. Reporting the two sub-sets separately lets the report distinguish in-domain improvement from out-of-domain behaviour. The set is held out from the SFT data and will be held out from the Section 4 preference prompts, so it is the only evaluation input the three models share.

The size and split (ten plus ten) were chosen as a deliberate middle: large enough that per-sub-set win-rates are not dominated by a single prompt, small enough that judging three model outputs on each with Gemma-4 stays cheap.

## 7. Planned stages (designed, not yet implemented)

The remaining notebook sections are intros only at this point. Their designs and decisions are settled; the code is the next work.

### 7.1 Preference dataset (Section 4)

Reinforcement learning from AI feedback. Recruiter prompts are drawn from the SFT validation records (held out from SFT training, and disjoint from the evaluation set), several candidate postings are sampled from the SFT model with temperature-based decoding, and the `atlm_teacher` Gemma-4 agent ranks them. The output is `(prompt, chosen, rejected)` triples written to `data/processed/ma2/preferences.jsonl`. Decisions taken: draw the prompts from the held-out validation records, on the order of a few hundred prompts; sample four candidates per prompt; have the judge rank the candidates in a single listwise call and take the best as chosen and the worst as rejected (one judge call per prompt, avoiding a pairwise blow-up); and judge against a rubric of faithfulness to the request, completeness and correct structure, professional and inclusive language, and absence of repetition or truncation. The judge-call code will follow the same `agent_server` calling pattern used by the ETL.

### 7.2 DPO training (Section 5)

The TRL `DPOTrainer`, with the SFT model as both the policy and the reference, a fresh LoRA on top of it, and the preference triples from Section 4. The output target is `outputs/ma2-360m-dpo/`. The beta (KL coefficient) and learning rate are the hyperparameters alignment is most sensitive to, so they will be stated and one variation considered.

### 7.3 Three-way evaluation execution (Section 6.2 onward)

The base SmolLM2-360M, the MA1-plus-SFT model, and the DPO-aligned model are run on the 20 frozen prompts. Automatic metrics (perplexity on a held-out set, and an LLM-as-judge win-rate) plus qualitative side-by-side examples are reported, with the in-distribution and out-of-distribution sub-sets kept separate. Note that the first leg, the base model, is the raw pretrained checkpoint with no MA1 LoRA, which is a different baseline from the merged MA1 model used in the Section 3.5 sanity check; the evaluation prose will make that explicit.

### 7.4 Limitations (Section 7)

The honest discussion of where alignment helped and where it regressed (verbosity, refusals, lost domain knowledge, sycophancy), which the brief weights explicitly. This is written after the results are in.

## 8. Notebook versioning and engineering notes

The working notebook started as `src/ma2/atlm_ma2_v1.ipynb`. After the SFT stage was implemented and run successfully, v1 was frozen as the record of that verified run, with its execution outputs, and work continued in a clean copy, `src/ma2/atlm_ma2_v2.ipynb` (25 cells), which is the canonical working notebook. The seven-section structure (Introduction, Starting point, SFT, Preference dataset, DPO, Evaluation, Limitations) maps onto the report's required sections.

Two engineering points came up during this stage and are worth carrying forward. First, the merge-cell CUDA fix described in Section 4: an environment variable set for a CPU-only operation must not be set process-wide in a notebook that also trains on the GPU. Second, the TRL API at version 1.5.1 uses `max_length` and `eval_strategy`; the older `max_seq_length` and `evaluation_strategy` names were verified to be gone before the run, which avoided wasting a GPU run on a config error.
