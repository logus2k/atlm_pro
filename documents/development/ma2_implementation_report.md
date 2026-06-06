# Mini-Assignment 2 - Implementation Report

This document records how Mini-Assignment 2 (alignment) was built: the development steps in order, what each step implements, the most relevant code, and the reasoning behind the options chosen. It is a development log, not the academic report. The academic report is the canonical notebook `src/ma2/atlm_ma2_v5_AC.ipynb`.

All MA2 stages are implemented. The pipeline has been executed end to end once under a Gemma-4-only judge ("Configuration B", v4 of the notebook), and is being re-executed under a cross-judge configuration ("Configuration C", v5_AC) at the time of this revision. Configuration C uses three independent local LLM families across the three pipeline roles - student, RLAIF judge, eval judge - to break the self-preference loop that Configuration B inherited from reusing Gemma as the data-preparation teacher *and* both judges. The decision to redo the run under Configuration C is documented in §10 below.

## 1. Goal and pipeline

The brief asks us to take the domain-adapted model from Mini-Assignment 1 and apply an alignment technique on top of it, walking the full post-training pipeline and reasoning about what each stage changes. The report cap is 5 pages.

The pipeline has four stages, the last two belonging to this assignment:

1. Pretraining: SmolLM2-360M, trained by its authors on general text.
2. Continued pretraining (Mini-Assignment 1): adapted to the job-postings domain by next-token training on raw IT descriptions. Produces a domain-fluent text completer.
3. Supervised fine-tuning (this assignment): teaches the completer to follow recruiter instructions, using (query, structured posting) pairs.
4. Preference alignment (this assignment): Direct Preference Optimization (DPO) with reinforcement learning from AI feedback (RLAIF), using a separate local LLM as judge.

The chosen alignment technique is DPO via RLAIF. DPO is the recommended default for a single-GPU budget (no separate reward model, no PPO instability), and RLAIF fits because the project already has capable local LLMs served by `agent_server`, so preferences can be generated rather than hand-annotated or pulled from a generic dataset.

In Configuration C three distinct model families fill the three pipeline roles to avoid shared-prior risk: SmolLM2-360M is the student, Nemotron Nano 4B serves the RLAIF listwise judge in Section 4.3, and Granite 3.3 2B serves the pairwise eval judge in Sections 6.4 and 6.7. Gemma 4 E2B remains the MA1 ETL teacher (`atlm_teacher`) that produced the SFT corpus but does not participate in any MA2 judge role.

## 2. Tooling and environment

The alignment stack is HuggingFace TRL 1.5.1, transformers 5.9.0, peft 0.19.1, and datasets, in the project virtual environment `.venv_atlm_pro` (Python 3.12), on a single RTX 4090 (24 GB). TRL was pinned at 1.5.1 with no version churn on transformers, torch, or peft. The seed is 42 throughout.

A note on the TRL API at this version, since it changed from older tutorials: `SFTConfig` takes `max_length` (not `max_seq_length`) and `eval_strategy` (not `evaluation_strategy`), and `SFTTrainer` takes `processing_class` and `peft_config`. The notebook code uses the current names.

External LLM calls go through `agent_server` (Docker, port 7701), an OpenAI-compatible gateway that fronts a llama.cpp router. Five local chat models are registered (`gemma-4`, `qwen3.5`, `smollm3`, `granite-3.3`, `nemotron`, `ministral`); only one is resident in VRAM at a time per the single-resident-model invariant documented in `~/env/assets/agent_server/documents/active_model_switching_sdk.md`. The notebook switches the active model via `POST /admin/api/active-model` immediately before each judge phase; the switch is detailed in §9.

## 3. Step 0 - Supervised fine-tuning data preparation

The SFT data did not exist as a labelled set; it was generated from the same raw postings used in Mini-Assignment 1, by an ETL agent. The agent, `atlm_teacher`, is a Gemma 4 model served by `agent_server`. Given a raw posting it returns three short recruiter queries plus a clean, structured job description in Markdown, or a skip marker for unusable input.

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

The result used for SFT is `data/processed/converted.jsonl`: 2,507 records, each with three queries and one structured job description, which fan out to roughly 7,500 (query, posting) training pairs.

Rationale for generating the data this way: the assignment needs instruction pairs in the job-postings domain. A generic instruction dataset would not match the domain; manual annotation of thousands of pairs is infeasible. Using a stronger model to convert raw postings into clean, consistently structured targets is exactly the RLAIF idea applied at the data-preparation stage, and it produces targets whose Markdown structure the SFT model can learn to imitate.

Note for §10: it is the reuse of Gemma 4 in this teacher role that motivates excluding Gemma from the MA2 judge roles under Configuration C. The teacher writes the queries and the structured postings the student learns from; reusing Gemma as judge afterwards would have the same model on two of the three pipeline roles, with the student's output distribution shaped by the teacher's stylistic prior.

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

This stage turns the domain-fluent completer into an instruction follower. It is implemented in Section 3 of the notebook and has been run. It has not been re-run under Configuration C - the SFT corpus and the resulting LoRA adapter are independent of the judge model and are reused verbatim.

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

The TRL trainer appends the EOS token to every formatted example, so the model learns when to stop. The tokenizer has no pad token, so it falls back to EOS; padding positions are masked out of the loss, so this is harmless. (A related observation: when later cells reload the model + tokenizer together, `transformers` may emit a one-line notice that `pad_token_id` was synced between the tokenizer and the model/generation configs. That sync is automatic and benign.)

### 5.5 Result and sanity check (Section 3.5)

The run completed in 18.8 minutes on the 4090. It trained 8.68M LoRA parameters (2.3 percent of the 370.5M total), over 6,771 train and 750 validation examples, reaching a final validation loss of 1.4955, a perplexity of 4.46. The 750 validation examples equal 250 held-out records times three queries, confirming the record-level split held.

The sanity check in Section 3.5 generates from the merged MA1 model and the SFT model side by side, with greedy decoding, on a few recruiter prompts. The expected and observed change is that the SFT model treats the request as an instruction and emits a structured posting with the right headings, where the pre-SFT model continues the text as free completion. This was confirmed qualitatively before freezing the run.

The SFT LoRA adapter is the input to both Section 4 (candidate sampling for the preference dataset) and Section 5 (DPO training as the policy initialisation and reference model).

## 6. Step 3 - Four-way evaluation prompt set (Section 6.1)

The evaluation prompt set was frozen early, before any further training, so the model states are measured on identical inputs and there is no opportunity to pick favourable prompts after the fact. The set is 20 recruiter queries, split into two ten-prompt sub-sets, persisted to `data/processed/ma2/eval_prompts.jsonl`:

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

The two sub-sets test different things. The ten in-distribution prompts are written in the style of the training data but are not part of it, so they are the fair measure of in-domain quality. The ten out-of-distribution prompts are fresh and probe generalisation: a junior bootcamp hire with no fixed stack, a principal staff engineer, niche stacks (Elixir and OTP, low-latency C++, Rust and WebAssembly), a freelance documentation writer, a soft-skills-heavy role, and a remote-async role. Reporting the two sub-sets separately lets the report distinguish in-domain improvement from out-of-domain behaviour. The set is held out from the SFT data and from the Section 4 preference prompts, so it is the only evaluation input the model states share.

The evaluation expanded from a three-way (base / SFT / DPO) to a four-way (base / SFT / DPO-b01 / DPO-b03) comparison once the beta sweep was introduced (§8), plus a focused DPO-b02 leg added in Section 6.7. The set size and split (ten plus ten) were chosen as a deliberate middle: large enough that per-sub-set win-rates are not dominated by a single prompt, small enough that the full 240-call win-rate sweep stays cheap on the judge model.

`ood-03` in this set is the Elixir-and-Phoenix prompt where DPO-b01 collapses into a 40-line repetition of "Container orchestration using Docker with ...". This case is used downstream as a hard sanity check on candidate judge models: a judge that misses the collapse is length- or position-biased and is not viable.

## 7. Step 4 - Preference dataset construction (Section 4)

This is where the alignment loop begins. The goal is a dataset of (prompt, chosen, rejected) triples that DPO can train on. RLAIF generates them by sampling several candidates from the SFT model and letting a separate, stronger LLM rank them. Three concrete decisions shape the implementation.

### 7.1 Prompts (Section 4.1)

The preference prompts are drawn from the SFT validation split rather than fresh authoring or the eval set. Drawing from the held-out SFT validation records guarantees the student has never trained on these prompts, so the candidate distribution is not memorised, and keeps the eval set strictly separate from any later DPO training signal. One query is picked per record using a separate seed from the record shuffle so the pick does not correlate with which records ended up in validation. The result is 250 preference prompts persisted to `data/processed/ma2/preference_prompts.jsonl`.

### 7.2 Candidate sampling (Section 4.2)

Four candidates are sampled per prompt with temperature 0.9 and top-p 0.95, `num_return_sequences=4` in a single batched `generate` call, max 500 new tokens. Sampling (rather than greedy decoding) is required: candidates have to differ enough that the judge can rank them, and greedy decoding would give four near-identical samples and produce no preference signal. The output is `data/processed/ma2/sft_candidates.jsonl`, idempotent on the (prompt_id, candidate_idx) key.

### 7.3 Judging (Section 4.3)

The judge model receives a recruiter request and four candidate postings labelled 1 to 4, and is asked for a listwise ranking that ends with the single line `RANKING: best=<id> worst=<id>`. Listwise is cheaper than pairwise: one judge call per prompt produces one (best, worst) pair, which is exactly what DPO needs; pairwise would multiply the call count by six.

The rubric is, in priority order: faithfulness to the request, structure and completeness (`## Summary`, `## Required Skills`, `## Responsibilities`, `## Requirements`, each non-empty), professional and inclusive language, and absence of repetition or truncation. The rubric lives in `agent_server`'s preset registry under the `atlm_rlaif_judge` agent (see §9), not inline in the notebook code cell; the cell only sends `{model: "atlm_rlaif_judge", messages: [{role: user, content: ...}]}`.

The judge call uses four worker threads against `agent_server` and is idempotent on `prompt_id`: prompts already in `data/processed/ma2/judge_ranks.jsonl` are skipped, so the run is resumable.

### 7.4 Preference triples (Section 4.4)

`assemble_preferences()` reads `judge_ranks.jsonl` and `sft_candidates.jsonl` and writes the (prompt, chosen, rejected) triples to `data/processed/ma2/preferences.jsonl`. The chosen is the candidate the judge ranked as best; the rejected is the candidate ranked as worst. The other two candidates are dropped; DPO trains on the extremes for the strongest gradient signal.

The Configuration B run (Gemma judge) produced 250 valid triples. The Configuration C run (Nemotron judge) is in progress; expected output is the same size, with the chosen and rejected texts re-derived from the Nemotron rankings.

## 8. Step 5 - DPO training (Section 5)

DPO updates the SFT policy to prefer the chosen response over the rejected one for each (prompt, chosen, rejected) triple, regularised by KL divergence to a reference policy. The reference is a frozen copy of the SFT model, so the policy only diverges as the preference signal warrants.

### 8.1 Training configuration (Section 5.2 - 5.4)

A fresh LoRA is trained on top of the SFT-merged base. Shape matches the SFT LoRA (`r=16`, `alpha=32`, `dropout=0.05`, same target modules) so the alignment delta is comparable. Effective batch size is the same as SFT (16). The DPO-specific settings:

```python
DPO_CFG = {
    "epochs":        5,
    "learning_rate": 5e-5,        # higher than SFT; LoRA-DPO needs more
    "betas":         [0.1, 0.2, 0.3],
}
```

The learning rate is `5e-5`, an order of magnitude higher than full-fine-tune DPO conventionally uses (`5e-6`). This was determined by iteration on Configuration B: at `5e-6` over 1 epoch DPO produced near-zero reward margins; at `5e-6` over 5 epochs still near-zero; `5e-5` over 5 epochs produced clean positive margins. LoRA-DPO is a small subspace of the full parameter space and needs a proportionally higher learning rate to move the reward function.

Three beta values are trained (`0.1`, `0.2`, `0.3`) to map the KL-coefficient sensitivity surface. Beta is the dominant alignment hyperparameter: lower beta lets the policy diverge further from the SFT reference (more aggressive alignment, more reward gain), higher beta keeps the policy close (more conservative, lower reward gain but lower risk of degeneration). All three are trained on the same preference set so they are directly comparable.

### 8.2 Outputs

The three runs write to `outputs/ma2-360m-dpo-{b01,b02,b03}/`, one LoRA adapter per beta. The Configuration B checkpoints have been moved to `outputs/ma2-360m-dpo-{b01,b02,b03}_gemma_run/` to preserve them; Configuration C overwrites the canonical paths.

The b02 leg (beta=0.2) was added partway through development as a sweet-spot probe after the Configuration B win-rate showed a monotonic preference for lower beta but with a documented failure mode at b01 (the `ood-03` repetition collapse). b02 is intended to be a "best of both" candidate; the Section 6.7 eval evaluates it against b01 and b03 on the 20 frozen prompts.

## 9. Calling agent_server (§9.1) and active-model switching (§9.2)

The MA2 judges are implemented as `agent_server` agent presets, following the same pattern as `atlm_teacher` and the existing `cv_rag_judge` and `noted_judge` agents. The infrastructure was already in place; what was added for MA2 are two presets (`atlm_rlaif_judge`, `atlm_eval_judge`) plus their two system-prompt files. These four files are version-controlled under `documents/development/agent_server_setup/` so they can be reinstalled, audited, or rolled back independently of the notebook.

### 9.1 Agent preset pattern (per `~/env/assets/agent_server/documents/how_to.md`)

Each agent is two files: a `<name>.agent.json` config (`name`, `system_prompt` path, `params_override`, `memory_policy`) and the prompt file it points at. At startup `agent_server` scans `data/agents/*.agent.json`, validates each strictly (a malformed file fails startup), and registers the preset by its `name`. After that the agent is callable two ways: by sending the agent name in the `model` field of `/v1/chat/completions`, or by fetching the resolved preset via `/v1/agents/<name>` and calling the underlying chat model directly.

The notebook uses the first form (Option A1 in the agent_server how_to):

```python
payload = {
    "model": "atlm_rlaif_judge",   # or "atlm_eval_judge" in Section 6
    "messages": [{"role": "user", "content": user_text}],
}
```

There is **no system message in the payload** - the rubric, temperature, `max_tokens`, and `chat_template_kwargs` come from the preset server-side. This is a deliberate inversion of how the code looked in Configuration B (v4 of the notebook), where the rubric was inlined as a `JUDGE_SYSTEM` string constant in the cell and sent on every call. Moving the rubric to the preset registry made the code cell simpler, made the rubric reviewable and reusable as a server-side artefact (the same way `atlm_teacher` lives), and gave the calibration battery a clean place to point at when documenting what each judge was actually told. The rubric text is also displayed as a fenced block in the relevant markdown cell so it remains visible in the PDF deliverable.

The two presets were installed via the admin API (Part 1B in `how_to.md`), which writes the files for you and hot-reloads the registry without a container restart:

```bash
curl -X POST http://localhost:7701/admin/api/agents -H 'Content-Type: application/json' -d '{
  "name": "atlm_rlaif_judge",
  "system_prompt": "<rubric text>",
  "params_override": {"max_tokens": 16384, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": true}},
  "memory_policy": "none"
}'
```

The `chat_template_kwargs` is per-family per SDK section 7b: `enable_thinking` for the gemma/qwen/smollm/nemotron families, `thinking` for granite, system-prompt-directive for the mistral family. The two MA2 judges set them according to the chat model each agent is intended to run on (`enable_thinking: true` for `atlm_rlaif_judge` running on Nemotron, `thinking: true` for `atlm_eval_judge` running on Granite).

### 9.2 Active-model switching (per `active_model_switching_sdk.md`)

`agent_server` keeps exactly one chat model resident in VRAM at a time (the "single-resident-model invariant"). Calling an agent preset always routes to whatever the active chat model is. So before invoking `atlm_rlaif_judge` the notebook must make Nemotron active; before invoking `atlm_eval_judge` it must make Granite active. The notebook does this through a single helper, `switch_active_model(model_id)`, defined once in cell `ma2s40helpers`:

```python
def switch_active_model(model_id):
    """Idempotent: no-op if already active. Otherwise POST to
    /admin/api/active-model, poll /v1/models until the new model serves,
    AND hold for at least SWITCH_MIN_WAIT_S=60 seconds total before
    returning. The 60s minimum is a client-side enforced hold the
    agent_server operator requires - even after /v1/models flips to the
    new model, downstream calls issued before the hold has elapsed can
    race the restart."""
```

Section 4.3 calls `switch_active_model("nemotron")` at the top of `judge_all()`; Sections 6.4 and 6.7 call `switch_active_model("granite-3.3")` at the top of their respective entry points. There are exactly two cold switches in a full Configuration C run, totalling ~2 minutes of wall time.

## 10. Step 6 - Calibration battery and the Configuration C decision (§§10.1 - 10.3)

### 10.1 Why a redo

Configuration B (the v4 notebook) ran with Gemma 4 in all three roles where an external LLM was called: as the MA1 ETL teacher (atlm_teacher, producing the queries and structured postings the student trained on), as the RLAIF listwise judge in Section 4.3, and as the pairwise eval judge in Section 6.4. That arrangement maximised cost efficiency (one model resident, no swap overhead) but stacked three shared-prior risks on top of each other: the SFT data inherited Gemma's stylistic priors via the teacher; the preference data inherited them again via the RLAIF judge ranking the student's outputs; and the win-rate measurement inherited them a third time via the same model scoring the result. Any systematic Gemma preference - for a particular phrasing, list shape, or sentence rhythm - would compound across the three roles, and the Section 7 limitations would have to caveat each headline number with that.

The redo splits the three roles across three distinct model families: Gemma stays only as the upstream MA1 teacher (whose output is already frozen in `converted.jsonl`), the RLAIF judge moves to a different family, and the eval judge moves to a third family different from both. This is what we call Configuration C.

### 10.2 The calibration battery

Picking the two replacement judges from `agent_server`'s five non-Gemma chat models (`qwen3.5`, `smollm3`, `granite-3.3`, `nemotron`, `ministral`) needed evidence, not a guess. Two probe scripts were written that exercise each candidate model on a small fixed sample of the actual MA2 judging task: 5 listwise RLAIF prompts (deterministic sample, seed 42, drawn from `preference_prompts.jsonl` filtered to records where Gemma had already ranked) and 5 pairwise eval pairs (hand-picked for coverage, with `ood-03` included as a hard sanity case - DPO-b01's repetition-loop collapse). Each pair is judged in both AB and BA orders, 20 calls per model, so the order-swap consistency metric the real Section 6.4 protocol depends on is directly probed.

The probes record, per call: parse OK, latency, completion tokens, `<think>` block size, predicted RANKING / VERDICT, agreement with Gemma's prior verdict on the same case (informational, not a target). At the end of all six runs (the five candidates plus Gemma as a control) a side-by-side report is generated at `documents/development/llm_models_performance.md`.

The result, as recorded in that report:

| Model | RLAIF parse | RLAIF think | RLAIF budget | Eval consistent | Eval Gemma agree | ood-03 caught | Eval budget |
|---|---|---|---|---|---|---|---|
| `gemma-4` (control) | 10/10 | yes (4,276 char avg) | 8.9 min | 9/10 | 7/7 | yes | 6.1 min |
| `nemotron` | **5/5** | **yes (4,898 char avg)** | **5.4 min** | 4/5 | 4/4 | yes | 3.1 min |
| `granite-3.3` | 1/5 | no | 1.8 min | **4/5** | **4/4** | **yes** | **1.4 min** |
| `smollm3` | 5/5 | no | 1.6 min | 3/5 | 2/2 | no | 0.9 min |
| `qwen3.5` | 1/5 | yes (50,807 chars - capped) | 91 min | 0/5 | n/a | no | 31 min |
| `ministral` | 5/5 | yes (36,252 chars, bimodal) | 48.8 min | 2/5 | 1/1 | yes | 3.2 min |

Disqualifiers (any single failure rules a model out of the role): RLAIF parse below 9/10, eval AB-BA consistency below 7/10, ood-03 missed. Tie-breakers: wall-time budget, agreement-with-Gemma (some disagreement is the whole point of cross-judge, but ~0 or ~10 are both red flags), think-chain availability for grounding the discussion.

### 10.3 Role assignment

- **RLAIF judge: `nemotron`.** Only non-Gemma candidate with 100 percent parse, real `<think>` reasoning (comparable in length to Gemma's), moderate Gemma agreement (3/5 best, 2/5 worst - exactly the cross-judge signal we want), and a tractable 5.4 min budget. Qwen blew the 16K-token cap; Granite and SmolLM3 emitted no `<think>` text; Ministral was bimodal in latency and capped on some prompts.
- **Eval judge: `granite-3.3`.** Matched Nemotron on every quality metric on the pairwise task (4/5 consistent, 4/4 Gemma agreement, caught `ood-03`) and is faster (1.4 min vs 3.1 min). Different family from Nemotron, so picking it preserves cross-judge independence between the RLAIF and eval roles. Granite emits no `<think>` tags, but its reasoning is plainly visible in the response text (confirmed by inspection on the `ood-03` case where it explicitly cites the repetition-loop unacceptability).
- **Excluded:** Gemma (MA1 teacher overlap), Qwen3.5 (RLAIF parse 20 percent and eval 0/5 consistency with extreme position bias), SmolLM3 (eval 60 percent consistency, missed `ood-03`, soft family overlap with the SmolLM2 student), Ministral (eval 40 percent consistency with the position-bias direction flipping pair-to-pair).

The decision plus the calibration table is reproduced verbatim in `documents/development/llm_models_performance.md`, and the per-model raw JSONs (one per probe, including each judge's full reasoning text) are in `documents/development/llm_calibration/`. After role assignment, the agent presets were installed (§9), the notebook code cells were rewritten to call the agent presets, an end-to-end validation script (`/tmp/validate_agent_presets.py`) was run that exercises the full path (switch → call → parse → inspect reasoning on `ood-03`), and the Configuration C run was started.

## 11. Step 7 - Three-way evaluation execution (Section 6)

The four model states (base SmolLM2-360M, MA1+SFT, MA2 DPO-b01, MA2 DPO-b03) are run on the 20 frozen prompts. Automatic metrics (perplexity on a held-out set, and the LLM-as-judge win-rate) plus qualitative side-by-side examples are reported, with the in-distribution and out-of-distribution sub-sets kept separate. Section 6.7 separately compares DPO-b02 against b01 and b03 on the same 20 prompts. Note that the first leg, the base model, is the raw pretrained checkpoint with no MA1 LoRA, which is a different baseline from the merged MA1 model used in the Section 3.5 sanity check; the evaluation prose makes that explicit.

Section 6.4 uses an order-swap protocol: every (model-A vs model-B, prompt) pair is judged twice with the candidates in opposite orders. Only when both orderings agree is the pair-prompt counted as a clean win; disagreement is reported as inconsistency rate (a diagnostic for how noisy the judge is). The rubric was tightened between the first Configuration B win-rate and the final one to add an explicit "length is not a quality indicator" instruction and promote anti-repetition to criterion #2; that corrected rubric is the one carried into `atlm_eval_judge`'s preset.

Configuration C results will be filled in here once the run completes.

## 12. Limitations (Section 7)

The honest discussion of where alignment helped and where it regressed (verbosity, refusals, lost domain knowledge, sycophancy), which the brief weights explicitly. Section 7 in the notebook holds three sub-sections:

- 7.1: What alignment changed (the base / SFT / DPO contrast across the four metrics).
- 7.2: Beta sensitivity in this run (the b01 / b02 / b03 contrast, including the substantive-content root-cause finding from reading the judge's `<think>` reasoning on prompts where simpler explanations failed).
- 7.3: Failure modes, limitations, and future work (the `ood-03` collapse at b01, the small evaluation set, hyperparameter coverage, and the methodological caveats - in Configuration B these were dominated by the single-judge self-preference circle; in Configuration C they are dominated by the small N and the limited beta-by-lr grid).

These sections will be rewritten against the Configuration C numbers once the run is in and the new judge reasoning has been inspected.

## 13. Notebook versioning and engineering notes

The working notebook progressed through five versions:

- `atlm_ma2_v1.ipynb`: SFT implemented and run; v1 was frozen as the record of the SFT run with its execution outputs.
- `atlm_ma2_v2.ipynb`: clean copy of v1 to continue work on the preference dataset and DPO sections.
- `atlm_ma2_v3.ipynb`: first end-to-end RLAIF run with Gemma as judge (Configuration A); revealed the initial DPO learning-rate problem (`5e-6` gave near-zero margins) and the original judge rubric's length bias.
- `atlm_ma2_v4_AC.ipynb`: 66 cells, the canonical Configuration B working notebook. SFT unchanged; preference dataset re-judged with the corrected rubric (anti-repetition + anti-length-bias); DPO retrained at three betas; Section 6 win-rate with order-swap protocol; Section 7 root-cause analysis grounded in Gemma's `<think>` text.
- `atlm_ma2_v5_AC.ipynb` (current canonical): clean copy of v4 with the agent-preset refactor: the helper cell `ma2s40helpers` defines `switch_active_model`, the three judge cells (`ma2s43code`, `ma2s64code`, `ma2s67code`) call the agent presets, the markdown cells (`ma2s1pipe`, `ma2s4`, `ma2s42`, `ma2s43`, `ma2s64`) describe the Configuration C judges, and the rubrics are shown as fenced blocks for the PDF deliverable. 68 cells total.

Engineering observations carried forward:

- **Merge-cell CUDA fix** (§4): an environment variable set for a CPU-only operation must not be set process-wide in a notebook that also trains on the GPU.
- **TRL 1.5.1 API names** (§2): `max_length` and `eval_strategy`, not `max_seq_length` and `evaluation_strategy`. Verified before each run.
- **DPO learning rate for LoRA** (§8.1): conventional `5e-6` is for full-fine-tune DPO; LoRA-DPO needs `1e-5` to `5e-5` to produce non-zero reward margins.
- **Judge `<think>` budget** (§7.3): Gemma's `<think>` chain on the listwise task averages 4,276 chars; the original `max_tokens=800` cap caused 99 percent parse failures on the first Configuration A judging run. Lifting to 16,384 fixed it and is now the standard for every judge call. Qwen's chain on the same task averages 50,807 chars (capped) - that was the disqualifying signal in the calibration battery.
- **Active-model switch hold** (§9.2): the SDK doc reports a 30-45 second switch latency, but `agent_server` clients must enforce an additional client-side minimum hold (~60 s total) before the next call. Polling `/v1/models` for `active: true` is necessary but not sufficient. The `switch_active_model` helper bakes both checks in.
- **Backup / restore discipline**: before launching Configuration C, the Configuration B run's artefacts were moved to `data/processed/ma2/gemma_run/` and `outputs/ma2-360m-dpo-{b01,b02,b03}_gemma_run/`. The judge log files (`judge_ranks.jsonl`, `winrate_calls.jsonl`) were truncated so the resume logic in the cells starts fresh; the SFT corpus and the `eval_generations/{base,sft}.jsonl` files were left untouched because they are independent of the judge model.

These five versions, plus the calibration report and the four `agent_server_setup/` files, are the full development trail of MA2.
