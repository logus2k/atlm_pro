# v6 post-run finalisation checklist

This document is the operational checklist for finalising
`src/ma2/atlm_ma2_v6_AC.ipynb` into a deliverable notebook after the
2026-06-06 top-to-bottom run completes. It is a working file (development
log, not a deliverable artefact), but the notebook prose pass should
follow it cell by cell.

Audience: future-me after compaction. The reading order is strict and
the constraints below are non-negotiable.

## 0. Constraints that override every decision below

1. **Read the actual data on disk before writing any prose.** No prose
   without a file path it derives from. The full incident log of what
   happens when this rule is ignored is in
   `~/.claude/projects/-home-logus-env-iscte-atlm-pro/memory/feedback-check-the-data.md`.
2. **Do not invent numbers.** Every numeric claim cites the file that
   produced it.
3. **Do not pattern-match Configuration B's voice onto v6 numbers.** The
   Configuration B prose was grounded in Gemma's judge, b01-on-ood-03
   collapse, "DPO-b01 wins decisively over SFT" framing. None of that
   carries forward verbatim. Granite (the v6 eval judge) produces a
   genuinely different picture on close pairs.
4. **Do not call partial evidence "direct evidence".** This was the
   LR=1e-4 mistake. If the recommendation rests on training-side metrics
   only and not on generations + perplexity + win-rate, say so explicitly.
5. **Do not call the 3-point beta probe a "sweep".** It is a beta
   sensitivity check. A real sweep would be a `(beta x LR)` grid with
   replication.
6. **Report cap is 10-15 pages, NOT 5.** The 5-page cap that appears in
   older memory is wrong for MA2.

## 1. Read the data in this order

After Section 6.4 finishes, before writing anything:

### 1a. Training-side telemetry
For each of `outputs/ma2-360m-dpo-{b01,b02,b03}/`:
- `summary.json`: `eval_loss`, `eval_mean_token_accuracy`,
  `eval_rewards/{chosen,rejected,accuracies,margins}`, `minutes`,
  `train_examples`, `val_examples`.
- `log_history.json`: trajectory of train loss vs eval loss across
  steps. Identify the train/eval gap at end. Where did eval margin
  saturate (or did it still rise at the last step)? Bigger sample of
  the trajectory: train_rwd_acc vs eval_rwd_acc. The Configuration C
  signature was train 0.97 vs eval 0.67 (heavy overfit). v6 with 1500
  triples vs 222 should narrow that gap.

### 1b. Generations
For each prompt id in `{ind-01, ind-05, ind-07, ind-08, ood-01,
ood-03, ood-07}` read all five sets of generations:
`data/processed/ma2/eval_generations/{base,sft,dpo-b01,dpo-b02,dpo-b03}.jsonl`.

Look explicitly for:
- The "first-in, first-out" hallucination phrase (the LR=1e-4 failure
  signature). If it appears, the v6 LR=5e-5 also tipped the policy off
  the deep end - investigate before continuing.
- Repetition loops (e.g. "Container orchestration using Docker"
  repeated).
- Role misreads (e.g. ood-07 producing a "Technical Support Specialist"
  posting when the request is for a developer who runs workshops; this
  was the Configuration C example of DPO improving over SFT).
- ood-03 should produce a clean Elixir / Phoenix posting in every DPO
  model (the Gemma-only Configuration B collapse here was Gemma-judge-
  specific, not low-beta-intrinsic).

### 1c. Judge reasoning
From `data/processed/ma2/winrate_calls.jsonl` pull 5 close-pair
verdicts where the judge committed consistently and 5 where it
flipped. Read the `raw` field. Does Granite's reasoning cite specific
rubric criteria (faithfulness, structure, repetition, language)? Or
does it hand-wave? This is what grounds Section 7's discussion of the
judge's discrimination floor.

### 1d. Perplexity
`data/processed/ma2/eval_perplexity.json` - five rows. Compare to
Configuration C baseline:
- base ~11.65 (unchanged)
- sft ~4.54 (unchanged - same SFT-merged base)
- dpo-b01 ~4.64 in Configuration C; was 8.15 in v6 LR=1e-4 attempt.
  v6 with LR=5e-5 should land in ~4.7-5.5 range.
- dpo-b02 ~4.61 in Configuration C
- dpo-b03 ~4.59 in Configuration C

If any DPO row > ~5.5 in v6, that's the LR=1e-4 disaster signature -
investigate before drawing conclusions.

### 1e. Win-rate
`data/processed/ma2/eval_winrate.json` - 10 pairs. For each:
- `wins_a`, `wins_b`, `inconsistent` counts (out of 20 prompts)
- `agreement_rate` (Granite's AB-BA consistency)

The interesting question: did the close-pair (sft vs dpo, dpo vs
dpo) agreement_rate move off Configuration C's 5-15% floor? If yes,
the larger preference signal moved the policy enough that Granite
can see real differences. If no, the judge discrimination ceiling is
structural to (granite, N=20).

## 2. Markdown cells to rewrite, in order

### ma2s56 (Section 5.6 - Training results table + hyperparameter sensitivity intro)
- Input: section 1a above.
- Write: one 3-column table (b01 / b02 / b03) with `eval_rewards/margins`,
  `eval_rewards/accuracies`, `eval_loss`, `eval_mean_token_accuracy`.
- One paragraph: whether the metrics rank monotonically with beta on
  the training-distribution preference set. State the train/eval gap
  (memorisation signal) without overclaiming.
- One paragraph: hyperparameter sensitivity narrative. Keep the LR-
  iteration history (5e-6 → 5e-5 finding from v3) and add: the LR=1e-4
  attempt produced visible fluency degradation despite higher training-
  side margin; we reverted to 5e-5 and the v6 numbers above are the
  result. This is the concrete sensitivity finding for the report.
- Cross-reference: see Section 7 for what the training-side metrics
  do and do not tell us about deployment quality.

### ma2s63 (Section 6.3 - Perplexity narrative)
- Input: section 1d.
- Write: short prose plus the 5-row table.
- Compare to Configuration C baseline numbers explicitly.
- State that DPO is *expected* to raise perplexity slightly because
  it shifts the policy off the SFT distribution; what matters is
  whether the shift is proportionate.

### ma2s65 (Section 6.5 - Qualitative side-by-side intro)
- The code (ma2s65code) already produces the display.
- Write: name the four `QUALITATIVE_IDS` the code uses, explain why
  they were chosen, and tell the reader what to look for in the
  displayed comparison (hallucinated phrases, repetition loops,
  role-faithfulness).

### ma2s66 (Section 6.6 - Behavioural)
- Input: the cell output produced by ma2s66code.
- Write: prose describing the behavioural pattern across 5 models.
- Mean output length, percentage of generations with all four required
  sections, the most-divergent prompts.
- Do NOT assert which model is "best" from these descriptive stats.

### ma2s68 (Section 6.8 - Synthesis)
- Input: all three pieces (perplexity, win-rate, behavioural).
- Write: where the three pieces of evidence agree, where they diverge.
- Frame around: the base loses to everything cleanly (Section 6.4
  obvious pairs); the interesting question is the close-pair regime.
- Did Granite's AB-BA consistency on close pairs move off the
  Configuration C 5-15% floor? State which case we're in based on
  actual eval_winrate.json numbers.

### ma2s71 (Section 7.1 - What alignment changed)
- The base / SFT / DPO contrast.
- Ground every assertion in win-rate numbers + qualitative examples
  from eval_generations/.
- The Configuration B prose framing this section as "DPO-b01 wins
  decisively" does not carry forward. The new framing depends on what
  the close-pair win-rate actually shows.

### ma2s72 (Section 7.2 - Beta sensitivity in this run)
- Call it a "beta sensitivity check" not a "sweep". Three points on
  one axis.
- What happened across (b01, b02, b03) on: perplexity, win-rate,
  behavioural stats.
- Discuss whether the trade-off direction is what we expected.
- If a low-end probe was added (b005), include it here as evidence
  about the cliff below b01.
- Do NOT claim "b02 sits between" as a finding (perplexity
  monotonicity is by definition; win-rate may not even reach
  significance to claim ordering).
- The LR-iteration history (5e-6 produced no signal, 5e-5 produced
  clean signal, 1e-4 produced fluency collapse) is the concrete
  hyperparameter-sensitivity story.

### ma2s73 (Section 7.3 - Failure modes, limitations, future work)
The substantial methodological chapter. With 10-15 pages we have
room for each of these as its own subsection or paragraph:
- The cross-judge protocol motivation and limitations. Granite's
  ~2B scale produces a discrimination floor on close pairs that
  surfaces as 85-95% AB-BA inconsistency in Configuration C; the
  v6 result either confirms or moves off that floor.
- The N=20 eval set bounds win-rate CIs widely on close
  comparisons.
- The 3-point one-axis beta probe vs a real (beta x LR) grid.
  Acknowledge what we would do with more compute.
- The 1500-triple full-ranking dataset is 6x the Configuration C
  signal but still derived from 250 prompts and 4 candidates -
  candidate diversity is still bounded.
- The Configuration A → B → C → v6 iteration history as a
  methodological learning (Gemma self-preference, judge discrimination
  floor). With space this is its own subsection.
- The LR=1e-4 incident as direct evidence the alignment-fluency
  trade-off is sharp at lower beta. The b01 hallucinations
  ("first-in, first-out") are a concrete failure case to cite.
- Honest cross-judge calibration story: the calibration battery in
  `documents/development/llm_models_performance.md` selected nemotron +
  granite from 6 candidates; this is part of the methodological story.

## 3. Cells that do NOT need rewriting

- ma2s0..ma2s4 (intro, MA1 merge, SFT)
- ma2s43 (Section 4.3 - judge agent setup)
- ma2s64 (Section 6.4 - win-rate setup)

These were rewritten correctly for v6 already.

## 4. Cheap follow-on probes worth considering

After the prose pass is done and BEFORE the delivery bundle:

- **b005 (beta=0.05)**: ~30 min training + ~5 min eval against the
  existing 5 models. The only point that would give the report new
  information (below the conventional 0.1 floor, tests where the
  fluency cliff lives). The LR=1e-4 disaster suggests the cliff is
  not far below b01.

If b005 is added, ma2s72 (Section 7.2) gains a 4th data point and the
prose changes accordingly.

## 5. Delivery bundle (after the prose pass)

Mirror `mp1/delivery/`:
- `ma2/delivery/atlm_ma2_groupc.ipynb` (the finalised notebook, all
  cell outputs preserved)
- `ma2/delivery/atlm_ma2_report_groupc.pdf` (PDF export)
- `ma2/delivery/README.md` (mirror mp1's README structure)
- `ma2/delivery/requirements.txt` (pinned subset)
- `ma2/delivery/src/` if any project-local source is needed

## 6. Memory references

After the prose pass is done, update memory:
- `~/.claude/projects/-home-logus-env-iscte-atlm-pro/memory/project-overview.md`
  - status moves from "v6_AC running" to "v6_AC delivered (date)"
- `~/.claude/projects/-home-logus-env-iscte-atlm-pro/memory/next-steps.md`
  - mark all post-run tasks complete
- If new lessons came out of the prose pass, add an incident to
  `feedback-check-the-data.md`

## 7. Backup paths preserved (do not touch)

- `outputs/ma2-360m-dpo-*_gemma_run/` (Configuration B v4)
- `outputs/ma2-360m-dpo-*_4cand_bestworst/` (Configuration C v5)
- `outputs/ma2-360m-dpo-*_phase_a_aborted/` (250 best/worst LR=1e-4 attempt #1)
- `outputs/ma2-360m-dpo-*_v6_lr1e-4_aborted/` (v6 LR=1e-4 attempt)
- `data/processed/ma2/sft_candidates.jsonl.pre-v6run-*` (pre-v6 SFT candidates)
- `data/processed/ma2/gemma_run/` (Configuration B data backup)
