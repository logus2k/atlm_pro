# LLM models performance — Configuration C calibration

This report compares candidate judge models on the two LLM-as-Judge roles in MA2: the listwise RLAIF judge of Section 4.3 (250 prompts in the real run) and the pairwise eval judge of Section 6.4 (240 calls in the real run). The numbers are produced by a small calibration battery (10 RLAIF prompts, 10 eval pairs run in both AB and BA orders) executed against each candidate model through `agent_server` after switching the active model via `POST /admin/api/active-model`. Both judges are invoked with thinking locked ON via `chat_template_kwargs` (SDK section 7b), so the comparison is apples-to-apples.

## Calibration design

- **RLAIF probe:** 10 preference prompts drawn deterministically (`random.seed(42)`) from `preference_prompts.jsonl`, filtered to records that have all four candidate generations and a Gemma ranking on disk. The same 10 prompts hit every candidate model.
- **Eval probe:** 10 hand-picked pairs covering all four pair types (`base/sft`, `sft/dpo`, close `dpo/dpo`), with `ood-03` included as a hard sanity case (DPO-b01 collapsed there into a 40-line repetition loop; a competent judge should pick DPO-b03). Each pair judged in both AB and BA orders, 20 calls total.
- **Switch:** before each candidate's run the script POSTs to `/admin/api/active-model` and polls `/v1/models` until that model is the sole `active: true` chat entry, then warms it with a tiny completion. Switch latency is recorded separately.
- **Same prompts as the real run:** both judges' `JUDGE_SYSTEM` strings are extracted live from the notebook (`ma2s43code` and `ma2s64code`) so any rubric drift between the probe and the real Section 4 / Section 6 cells is impossible by construction.

## RLAIF judge probe (Section 4.3 role)

| Metric | gemma-4 | granite-3.3 | ministral | nemotron | qwen3.5 | smollm3 |
|---|---|---|---|---|---|---|
| Parse rate | 10/10 | 1/5 | 5/5 | 5/5 | 1/5 | 5/5 |
| Mean latency (s) | 8.6 | 1.7 | 46.8 | 5.2 | 87.3 | 1.5 |
| Median latency | 8.5 | 1.8 | 7.4 | 5.9 | 100.4 | 1.8 |
| Min / Max latency | 7.5 / 10.6 | 1.4 / 1.9 | 5.4 / 108.2 | 3.5 / 6.9 | 34.8 / 101.2 | 0.7 / 2.2 |
| Mean completion tokens | 1261.0 | 312.0 | 7169.0 | 1105.0 | 14267.0 | 389.0 |
| Mean think chars | 4276.0 | 0.0 | 36252.0 | 4898.0 | 50807.0 | 0.0 |
| Uses <think> | 10/10 | 0/5 | 5/5 | 5/5 | 5/5 | 0/5 |
| Agree Gemma (best) | 10/10 | 0/5 | 3/5 | 3/5 | 0/5 | 2/5 |
| Agree Gemma (worst) | 6/10 | 0/5 | 2/5 | 2/5 | 1/5 | 1/5 |
| **Section 4 budget @4w (min)** | **8.9** | **1.8** | **48.8** | **5.4** | **91.0** | **1.6** |
| Switch latency (s) | 0.0 | 36.5 | 36.5 | 36.5 | 0.0 | 36.5 |

## Eval judge probe (Section 6.4 role)

| Metric | gemma-4 | granite-3.3 | ministral | nemotron | qwen3.5 | smollm3 |
|---|---|---|---|---|---|---|
| Pairs / calls | 10 / 20 | 5 / 10 | 5 / 10 | 5 / 10 | 5 / 10 | 5 / 10 |
| Parse-fail pairs | 0/10 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| Consistent pairs | 9/10 | 4/5 | 2/5 | 4/5 | 0/5 | 3/5 |
| Inconsistent pairs (order-swap) | 1/10 | 1/5 | 3/5 | 1/5 | 5/5 | 2/5 |
| AB-BA consistency | 90% | 80% | 40% | 80% | 0% | 60% |
| Mean latency (s) | 6.1 | 1.4 | 3.2 | 3.1 | 31.3 | 0.9 |
| Median latency | 6.1 | 1.4 | 3.4 | 3.2 | 12.7 | 0.9 |
| Min / Max latency | 4.4 / 7.9 | 1.1 / 2.0 | 1.9 / 5.0 | 1.9 / 4.1 | 6.5 / 100.3 | 0.5 / 1.6 |
| Mean completion tokens | 902.0 | 258.0 | 540.0 | 635.0 | 5163.0 | 222.0 |
| Uses <think> | 20/20 | 0/10 | 10/10 | 10/10 | 10/10 | 0/10 |
| Agree Gemma (pair-level) | 7/7 | 4/4 | 1/1 | 4/4 | n/a | 2/2 |
| ood-03 catches b01 collapse | yes | yes | yes | yes | no (None) | no (None) |
| **Section 6 budget @4w (min)** | **6.1** | **1.4** | **3.2** | **3.1** | **31.3** | **0.9** |
| Switch latency (s) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## How to read this report

**Disqualifiers** (any single failure rules a model out of the role):
- RLAIF parse rate below 9/10 means the model does not reliably emit the `RANKING: best=<id> worst=<id>` terminator; the Section 4 run would lose 25 or more of the 250 prompts.
- Eval AB-BA consistency below 7/10 means the model reads position, not content; the win-rate it produces is noise.
- ood-03 not catching the DPO-b01 collapse is a sanity-check failure on a case where the right answer is obvious; the judge is probably length-biased.

**Tie-breakers** (relevant once disqualifiers are clear):
- Wall-time budgets project Section 4 and Section 6 at four concurrent workers (matching `ma2s43code`'s `JUDGE_WORKERS = 4`). Larger numbers mean a longer real run, not a worse judge; they matter for planning, not for correctness.
- Agreement with Gemma is informational: full agreement signals the cross-judge experiment is moot (no new information against the existing baseline). Some disagreement is the whole point of the cross-judge protocol; the raw text on the disagreement cases is the place to decide whose call is more defensible.
- Think-chain length scales completion tokens and per-call cost. A judge that thinks 5x longer than another for the same verdict is paying for marginal accuracy you can measure here.

**Role-assignment heuristic** (from earlier analysis):
- Stronger model goes to the **eval judge** because its outputs are the report's headline win-rates; there is no downstream aggregation to smooth a weaker judge.
- Weaker model goes to the **RLAIF judge** because the 250-sample aggregation and DPO's documented tolerance to ~30 percent label noise absorb random noise. This only holds if the calibration shows no systematic bias on that model.

## Per-model notes

### `gemma-4`

- **RLAIF:** parse 10/10, mean latency 8.6 s, <think> 10/10, Gemma agreement 10/10 (best) and 6/10 (worst). Full 250-prompt run projects to 8.9 min at 4 workers.

- **Eval:** consistent 9/10, AB-BA 90%, parse fails 0/10, mean latency 6.1 s. ood-03 collapse: caught (winner=dpo-b03). Full 240-call run projects to 6.1 min at 4 workers.


### `granite-3.3`

- **RLAIF:** parse 1/5, mean latency 1.7 s, <think> 0/5, Gemma agreement 0/5 (best) and 0/5 (worst). Full 250-prompt run projects to 1.8 min at 4 workers.

- **Eval:** consistent 4/5, AB-BA 80%, parse fails 0/5, mean latency 1.4 s. ood-03 collapse: caught (winner=dpo-b03). Full 240-call run projects to 1.4 min at 4 workers.


### `ministral`

- **RLAIF:** parse 5/5, mean latency 46.8 s, <think> 5/5, Gemma agreement 3/5 (best) and 2/5 (worst). Full 250-prompt run projects to 48.8 min at 4 workers.

- **Eval:** consistent 2/5, AB-BA 40%, parse fails 0/5, mean latency 3.2 s. ood-03 collapse: caught (winner=dpo-b03). Full 240-call run projects to 3.2 min at 4 workers.


### `nemotron`

- **RLAIF:** parse 5/5, mean latency 5.2 s, <think> 5/5, Gemma agreement 3/5 (best) and 2/5 (worst). Full 250-prompt run projects to 5.4 min at 4 workers.

- **Eval:** consistent 4/5, AB-BA 80%, parse fails 0/5, mean latency 3.1 s. ood-03 collapse: caught (winner=dpo-b03). Full 240-call run projects to 3.1 min at 4 workers.


### `qwen3.5`

- **RLAIF:** parse 1/5, mean latency 87.3 s, <think> 5/5, Gemma agreement 0/5 (best) and 1/5 (worst). Full 250-prompt run projects to 91.0 min at 4 workers.

- **Eval:** consistent 0/5, AB-BA 0%, parse fails 0/5, mean latency 31.3 s. ood-03 collapse: missed (winner=None). Full 240-call run projects to 31.3 min at 4 workers.


### `smollm3`

- **RLAIF:** parse 5/5, mean latency 1.5 s, <think> 0/5, Gemma agreement 2/5 (best) and 1/5 (worst). Full 250-prompt run projects to 1.6 min at 4 workers.

- **Eval:** consistent 3/5, AB-BA 60%, parse fails 0/5, mean latency 0.9 s. ood-03 collapse: missed (winner=None). Full 240-call run projects to 0.9 min at 4 workers.


## Role assignment decision

- **RLAIF judge (Section 4.3): `nemotron`.** Rationale: the only non-Gemma candidate that combines a 100 percent parse rate (5/5), a real `<think>` chain on every call (mean 4,898 chars, comparable to Gemma's 4,276), a moderate Gemma-agreement signal (3/5 best, 2/5 worst — disagreement is the cross-judge protocol working, not failing) and a tractable budget (5.4 min for the full 250-prompt Section 4 run at four workers). The two other 100 percent-parse non-Gemma candidates fail elsewhere: `smollm3` emits zero `<think>` content despite the kwarg and would deny Section 7 any reasoning text to ground the discussion in, and `ministral` is bimodal (median 7.4 s but two of five calls hit the 16,384-token cap at ~108 s each), projecting a 48.8 min Section 4 run.

- **Eval judge (Section 6.4): `granite-3.3`.** Rationale: 5/5 parse, 4/5 AB-BA consistent, 4/4 agreement with Gemma on the comparable pairs, correctly catches the DPO-b01 collapse on ood-03, and finishes the full 240-call Section 6 run in 1.4 min. It is tied on every quality metric with nemotron on this probe but is faster, smaller, and from a different family, and using it here preserves cross-judge independence between the RLAIF role (`nemotron`) and the eval role (`granite-3.3`). The known cost is that `granite-3.3` emits zero `<think>` content in this deployment despite the `thinking: true` kwarg, so any reasoning-text grounding for Section 7 will need a small post-hoc pass with a `<think>`-emitting model on the specific controversial verdicts.

- **Models excluded:**
  - `gemma-4`: already the MA1 teacher (`atlm_teacher` produced `converted.jsonl`). Reusing it as an MA2 judge would put the same model on two of the three project roles and weaken the independence story Section 7 wants to tell. Best-in-class on every probe metric here, but its job in MA2 is to be the upstream control, not a judge.
  - `qwen3.5`: 1/5 RLAIF parse — its think chains for the listwise task systematically exceed the 16,384-token cap (mean think chars 50,807, vs gemma's 4,276); 0/5 eval AB-BA consistency, with extreme position bias (picked the first-presented candidate in 10/10 calls); missed the ood-03 collapse. Disqualified for both roles.
  - `smollm3`: 60 percent eval AB-BA consistency (below the 70 percent threshold), missed the ood-03 collapse, 0 `<think>` blocks emitted despite `enable_thinking: true`. Also has soft family overlap with the SmolLM2-360M student. Borderline-fast on RLAIF (1.6 min budget, 100 percent parse) but the lack of reasoning text and the eval failures rule it out.
  - `ministral`: 40 percent eval AB-BA consistency, with the position-bias direction flipping pair-to-pair (last-presented in pair 1, first-presented in pair 2) — not a stable bias the order-swap protocol could correct for, but unreliable judgment. Caught ood-03 but only by accident given its overall inconsistency. RLAIF run is feasible (100 percent parse) but slow and bimodal (48.8 min budget, two of five calls hit the 16,384-token cap).

### Methodological footnotes

- All probes ran with thinking ON, locked via `chat_template_kwargs.enable_thinking: true` (gemma-4, qwen3.5, smollm3, nemotron), `chat_template_kwargs.thinking: true` (granite-3.3), or — for the mistral family which has no kwarg per SDK section 7b — a `[THINK]...[/THINK]` directive appended to the system prompt (ministral). The ministral system prompt is therefore strictly larger than the others' by the length of that directive; this is plumbing to enable thinking, not a rubric change.
- `smollm3` and `granite-3.3` both emit zero `<think>` blocks across every call in this deployment despite the kwarg being set. The text inside the response on those models is presumably reasoning that the chat template did not wrap. This is consistent across both the RLAIF and eval probes for each model, so the behaviour is the model's not the prompt's.
- The Section 4 / Section 6 budget projections multiply the per-call mean latency by the real call count and divide by four concurrent workers (matching `JUDGE_WORKERS = 4` in `ma2s43code` and `ma2s64code`). They ignore agent_server queueing overhead, which the original Gemma run (25 min actual) suggests adds roughly 30 to 50 percent on top.
- The probe is N = 5 prompts (RLAIF) and N = 5 pairs (eval) for the five non-baseline models. Gemma runs at N = 10 each because it executed earlier when the per-call latency was unknown. The X/N format in every table cell preserves the difference; the metrics are still comparable because Gemma is a control, not a candidate.

## Source data

Per-model raw JSONs (full per-call records including each judge's raw text) live under [`llm_calibration/`](llm_calibration/):

- `gemma-4`: [`llm_calibration/rlaif_gemma-4.json`](llm_calibration/rlaif_gemma-4.json), [`llm_calibration/eval_gemma-4.json`](llm_calibration/eval_gemma-4.json)
- `granite-3.3`: [`llm_calibration/rlaif_granite-3.3.json`](llm_calibration/rlaif_granite-3.3.json), [`llm_calibration/eval_granite-3.3.json`](llm_calibration/eval_granite-3.3.json)
- `ministral`: [`llm_calibration/rlaif_ministral.json`](llm_calibration/rlaif_ministral.json), [`llm_calibration/eval_ministral.json`](llm_calibration/eval_ministral.json)
- `nemotron`: [`llm_calibration/rlaif_nemotron.json`](llm_calibration/rlaif_nemotron.json), [`llm_calibration/eval_nemotron.json`](llm_calibration/eval_nemotron.json)
- `qwen3.5`: [`llm_calibration/rlaif_qwen3.5.json`](llm_calibration/rlaif_qwen3.5.json), [`llm_calibration/eval_qwen3.5.json`](llm_calibration/eval_qwen3.5.json)
- `smollm3`: [`llm_calibration/rlaif_smollm3.json`](llm_calibration/rlaif_smollm3.json), [`llm_calibration/eval_smollm3.json`](llm_calibration/eval_smollm3.json)
