# `main (27).pdf` - Issues and validation findings

Comprehensive validation of `src/ma2/delivery/main (27).pdf` against the
on-disk truth (cell outputs in `src/ma2/atlm_ma2_v6_AC.ipynb`, the JSON files
under `data/processed/ma2/`, and the `summary.json` files under each
`outputs/ma2-360m-*/`). Every numerical claim listed below has been
cross-checked against the actual data rather than against the notebook
markdown.

This document is the correction guide. Each fix entry names the page,
section, the verbatim wrong text, the verified correct value, and the
recommended replacement.

---

## A. Issues that require correction

### A1. Section 6.3 LLM-as-Judge Win-Rate, page 10 - wrong count and wrong "no SFT wins" claim

The report says:
> "Second, on the SFT vs. DPO pairs, DPO wins every consistent verdict across all four betas (0 SFT wins in 25 consistent pair-prompt verdicts), with zero losses."

On-disk reality from `data/processed/ma2/eval_winrate.json` (the same data
the report's Table 8 displays correctly):

| pair | DPO wins | SFT wins | consistent total |
|---|---|---|---|
| SFT vs DPO-b005 | 7 | **2** | 9 |
| SFT vs DPO-b01 | 8 | 0 | 8 |
| SFT vs DPO-b02 | 6 | 0 | 6 |
| SFT vs DPO-b03 | 4 | 0 | 4 |
| **all four betas** | **25** | **2** | **27** |
| **three in-range betas only** | **18** | **0** | **18** |

The "25 consistent verdicts" number conflates "DPO wins" (25) with
"consistent verdicts" (27). The "0 SFT wins" claim ignores the two SFT wins
on the b005 leg, which Table 8 itself shows.

The §6.5 Synthesis phrasing on page 11 is correct: "DPO wins 18 of 18
consistent pair-prompt comparisons across the three SFT-vs-DPO pairs, zero
losses". §6.3 should match that phrasing.

**Recommended replacement** for the §6.3 sentence:
> "Second, on the SFT vs. DPO pairs, DPO wins every consistent verdict on
> the three in-range betas (18 of 18 across the SFT-vs-DPO-b01, b02, b03
> pairs, zero losses). On the SFT vs. DPO-b005 pair the signal weakens
> (7 DPO wins, 2 SFT wins on consistent verdicts), the first hint of the
> beta=0.05 cliff that perplexity flags more sharply in Section 6.2."

### A2. Section 6.5 Synthesis, page 12 - "no structural regression relative to SFT" overclaims for DPO-b01

The report says, of the deliverable policy:
> "The deliverable policy is DPO-b01 ($\beta = 0.10$, learning rate $5 \times 10^{-5}$): it beats the base 18-0 with 90% judge agreement, beats SFT 8-0 in consistent verdicts, and shows no fluency collapse or structural regression relative to SFT."

Verified directly from `data/processed/ma2/eval_generations/*.jsonl` (counts
match Table 9 of the report exactly):

| model | all 4 sections | avg sections | mean chars |
|---|---|---|---|
| SFT | 3/20 | 2.80 | 1567 |
| DPO-b01 | 1/20 | 2.50 | 1458 |

So DPO-b01 lands all four required Markdown headings on 1 of 20 outputs
versus SFT's 3 of 20, and averages 2.50 sections per generation versus
SFT's 2.80. That is a small but measurable structural regression, not
"none". The cause is the `repetition_penalty=1.3` restoration triggering
earlier EOS on some prompts before the model reaches `## Requirements`. The
trade-off is honest methodology (decoder bug fixed at the cost of a few
truncated tails), not a flaw to hide.

**Recommended replacement** for the structural sub-claim in §6.5:
> "...beats SFT 8-0 in consistent verdicts, shows no fluency collapse,
> and produces structural completeness comparable to SFT (1/20 vs 3/20
> all-four-section generations; 2.50 vs 2.80 average sections per
> generation). The small structural decrement is a known cost of
> restoring `repetition_penalty=1.3`: the kwarg occasionally triggers an
> early EOS before the model reaches `## Requirements`. Section 7
> documents this trade-off."

It is also worth adding one sentence in §7.1 or §7.4 making the rep_penalty
trade-off explicit ("the same kwarg that eliminates the token-loop
catastrophes also costs a few percent on structural completeness, by
triggering earlier EOS on prompts where the model would otherwise have
filled to max_new_tokens"). This is a real finding the report could keep.

### A3. Notebook §6.6 markdown and §7.1 markdown carry the same overclaim - needs syncing with the report

The notebook cells `ma2s66` (§6.6 Behavioural Statistics) and `ma2s71`
(§7.1 What alignment changed) currently say:

> "All four DPO models produce the four required Markdown headings on
> every one of the 20 evaluation generations."

This is wrong by the on-disk count (verified directly in
`data/processed/ma2/eval_generations/*.jsonl`):

| model | all 4 sections / 20 |
|---|---|
| DPO-b005 | 0 |
| DPO-b01 | 1 |
| DPO-b02 | 0 |
| DPO-b03 | 3 |

This is a notebook-side correction (not a report-side one), but if the
notebook is read as part of the deliverable the contradiction with the
report's Table 9 surfaces. Both notebook cells should be rewritten to
match the truth.

---

## B. Minor descriptive nuances (worth tightening, not strictly wrong)

### B1. Section 7.1 page 12 - "42 consecutive repetitions of SQS as bullet items"

The report says:
> "DPO-$\beta$=0.20 on the cloud-engineer prompt ind-07 emitted 42
> consecutive repetitions of the token SQS as bullet items in the
> Required Skills section"

Verified from `data/processed/ma2/v6_no_rep_penalty/eval_generations/dpo-b02.jsonl`:
the token SQS appears 171 times in the generation, but as a comma-separated
list within a single Required Skills bullet, not as 42 separate bullet items.
The "42" originates from the diagnostic 4-gram scanner counting "SQS, SQS,
SQS, SQS" appearing 42 times. That scan ran on the post-tokenisation surface
form and counted a sliding-window 4-gram.

**Recommended tightening** (preserves the finding, fixes the geometry):
> "DPO-$\beta$=0.20 on the cloud-engineer prompt ind-07 emitted a
> Required Skills bullet in which the AWS service token SQS repeated
> 171 times in a comma-separated list before EOS; DPO-$\beta$=0.30
> on the same prompt emitted an analogous degeneration with the ECS
> token, repeated as growing chains across more than 25 bullet items."

Or, if the "x consecutive bullet items" framing is desired, switch to b03's
ECS case which actually does manifest as escalating bullet items
("- ECS Pods", "- ECS ECS Pods", "- ECS ECS ECS Pods", ...).

### B2. Section 5.4, page 8 - "more epochs caused train reward accuracy to saturate at 1.00"

The report says:
> "Five epochs over 1,302 triples was determined empirically: fewer
> epochs left reward margins near zero; more caused the train reward
> accuracy to saturate at 1.00 with no corresponding deployment-side
> benefit (see Section 7)."

Verified directly from `outputs/ma2-360m-dpo-*/log_history.json`: training
reward accuracy already reaches 1.00 at 5 epochs across all four betas.
The "more epochs would saturate" comparative implies an experiment past 5
epochs was run; no such experiment is documented in the development log or
the on-disk artefacts. The saturation is already at 5 epochs.

**Recommended tightening:**
> "Five epochs over 1,302 triples was determined empirically: fewer
> epochs left reward margins near zero; at five epochs the training
> reward accuracy already saturates at 1.00 across all four betas, so
> running longer would not help and risks fluency degradation (see
> Section 7)."

---

## C. Validated correct (no action needed)

Every claim below was cross-checked against the on-disk data and verified
exact. Listed for the user's confidence:

- **Section 2.2 SFT corpus**: 2,507 raw postings (disk: 2507); 7,521
  query-posting pairs (disk: 7521); 90/10 record-level split into 2,257
  training records and 250 validation records (disk: exact match);
  fanned out to 6,771 training examples and 750 validation examples.
- **Section 3.3 Table 3 (SFT outcomes)**: every value matches the
  `outputs/ma2-360m-sft/summary.json` on-disk file: 8,683,520 trainable
  parameters, 2.34% of 370.5M total, 6,771 training examples, 750
  validation examples, final validation loss 1.4955, perplexity 4.46,
  wall-clock 17.4 minutes.
- **Section 4.3 Table 4 (cross-judge calibration)**: every RLAIF-parse,
  eval-consistent, RLAIF-budget and eval-budget value matches
  `documents/development/llm_models_performance.md`. The disqualifier
  decisions (gemma-4 control, nemotron RLAIF, granite-3.3 eval, others
  rejected) match the source.
- **Section 4.4 RLAIF results**: 241 of 250 prompts ranked successfully
  (disk: `data/processed/ma2/judge_ranks.jsonl` has 241 lines); 9 parse
  failures (250 - 241 = 9); 1,446 preference triples (disk:
  `data/processed/ma2/preferences.jsonl` has 1446 lines); 241 x 6 = 1446
  arithmetic verified; 6.5x expansion over Configurations B and C's 222
  best-worst triples verified.
- **Section 5.2 Table 5 (DPO config)**: 1,302 training triples + 144
  held-out evaluation triples = 1,446 total (matches every
  `outputs/ma2-360m-dpo-*/summary.json` exactly).
- **Section 5.4 Table 6 (DPO training-side metrics)**: every value for
  every beta matches the on-disk `summary.json` to three decimal places
  (eval_loss, eval reward accuracy, eval reward margin, eval entropy,
  eval mean token accuracy, wall-clock minutes).
- **Section 6.2 Table 7 (perplexity)**: every value matches
  `data/processed/ma2/eval_perplexity.json` exactly. The 47% perplexity
  jump from b01 (6.01) to b005 (8.84) is verified arithmetic
  (8.84/6.01 - 1 = 47.1%).
- **Section 6.3 Table 8 (win-rate matrix)**: every (A wins, B wins,
  inconsistent, agreement, winner) row matches
  `data/processed/ma2/eval_winrate.json` exactly. The base-vs-aligned
  65-90% AB-BA agreement range and the close-pair 15-40% range are
  correctly summarised.
- **Section 6.4 Table 9 (behavioural)**: every "mean chars", "all 4
  sections" and "avg sections" value matches a direct re-scan of the
  generations in `data/processed/ma2/eval_generations/*.jsonl`. The
  table itself is verified correct; the synthesis prose in §6.5 is what
  overstates (see issue A2).
- **Section 6.5 Synthesis 40% length drop claim**: b005 mean 869 chars
  vs b01 mean 1458 chars is a 40.4% drop, matches the "40%" report
  number.
- **Section 7.1 SFT corpus phrase scan**: "Container orchestration
  using" appears exactly 7 times across the 2,507 records in
  `data/processed/converted.jsonl`, with at most 1 occurrence per record.
  The "exactly seven times, once per record at most" claim is verified.
- **Section 7.1 Configuration B ood-03 collapse claim**: "Container
  orchestration" appears 51 times in the dpo-b01 ood-03 generation under
  `data/processed/ma2/gemma_run/eval_generations/dpo-b01.jsonl`. The
  "51-item repetition loop" claim is verified.
- **Section 7.1 effect of restoring rep_penalty**: confirmed that the
  current `data/processed/ma2/eval_generations/dpo-b02.jsonl` ind-07
  generation contains zero SQS tokens (loop eliminated), and dpo-b03
  ind-07 contains exactly 1 ECS occurrence (down from 273 in the
  pre-fix run). Token-loop catastrophes fully suppressed.
- **Cover, abstract, conclusion arithmetic**: the "18 of 18 consistent
  judge verdicts" framing for b01 against base, the "8-0 on consistent
  close-pair verdicts" framing for b01 against SFT, the "47% perplexity
  cliff", the "40% drop in mean output length" are all internally
  consistent and validated against on-disk data.

---

## D. Action checklist

1. [ ] Fix A1 (Section 6.3 sentence on page 10) in the LaTeX source.
2. [ ] Fix A2 (Section 6.5 Synthesis page 12, structural-regression
       claim) in the LaTeX source, optionally add one sentence in §7.1 or
       §7.4 about the rep_penalty trade-off.
3. [ ] Fix A3 (notebook cells `ma2s66` and `ma2s71` markdown) so the
       deliverable does not carry the same overclaim if anyone reads
       the notebook end to end.
4. [ ] Optional: tighten B1 and B2 (descriptive nuances around the SQS /
       ECS counts and around the 5-epoch saturation phrasing).
5. [ ] Cover replacement (user handling).

After 1-3 are applied, every numerical claim in the report and every
qualitative claim about the on-disk outputs is consistent with the
data on disk.

---

## E. Validation of the second LLM reviewer's feedback

The following items were submitted by an external LLM reviewer after the
first validation pass. Each is checked against the actual PDF text and
the on-disk data.

### E1. "DPO-6005, DPO-601, DPO-602, DPO-603 in Tables 8 & 9" - **FALSE**

The reviewer claims the model labels in Tables 8 and 9 contain
typographical errors of the form `DPO-6005`, `DPO-601`, etc., suggesting a
LaTeX rendering bug that swapped lowercase `b` for the digit `6`.

Verified by extracting the raw PDF text with `pdftotext`:

```
Base vs. DPO-b005           2    13    5    75%   DPO-b005
Base vs. DPO-b01            0    18    2    90%   DPO-b01
Base vs. DPO-b02            1    14    5    75%   DPO-b02
Base vs. DPO-b03            1    12    7    65%   DPO-b03
SFT vs. DPO-b005            2     7   11    45%   DPO-b005
... and so on for all 15 rows of Table 8.
```

The PDF labels are `DPO-b005`, `DPO-b01`, `DPO-b02`, `DPO-b03` everywhere.
No "6" anywhere. This reviewer-claim is a character-recognition
misperception by the reviewer LLM, not a real defect. No action required.

### E2. "Table 4 ood-03 column blanks for nemotron, granite-3.3, ministral" - **TRUE observation, valid clarity improvement**

Verified against the PDF on page 6: Table 4's `ood-03` column shows an
explicit `x` for smollm3 and qwen3.5 (the two models that missed the
collapse), and visually empty cells for gemma-4, nemotron, granite-3.3
and ministral.

The intended semantic is "empty cell = caught the collapse", "x = missed
it". A reader scanning quickly may assume "empty" means "not tested" or
"data unavailable" rather than "passed". Filling the empty cells with an
explicit checkmark would remove the ambiguity.

**Recommended fix**: replace the four empty cells in Table 4's `ood-03`
column with a checkmark (✓) or the word "yes". Smollm3 and qwen3.5 keep
the `x`.

### E3. "SFT loss masking: prompt span not masked" - **Valid critical-discussion expansion, not an error**

The reviewer notes that §3.2 computes the SFT loss over the entire
formatted sequence (preamble + request + response) rather than masking
the prompt span, and suggests this is a methodological weakness worth
flagging in Section 7.

This is not an error: the report already acknowledges the choice in §3.2
("the loss is computed over the entire formatted sequence... Masking the
prompt span would be the theoretically cleaner choice; for a model of
this size the difference is minor and the simpler configuration was
retained"). The reviewer's suggestion is that this acknowledgement could
move or be expanded in Section 7.

**Recommended action (optional)**: leave §3.2 as is, and add one bullet
to Section 7.4 or to §7.6 future work along the lines of: "the
unmasked-prompt-span SFT loss formulation, retained for simplicity,
introduces a small optimisation overhead and a mild exposure-bias
exposure on the preamble tokens that masking would avoid; the cost was
not measured here". This is enhancement rather than correction.

### E4. "Dataset nomenclature: SFT validation vs DPO validation" - **Valid clarity improvement**

The reviewer notes that the report references two distinct held-out sets
without always disambiguating them:

- **DPO preference validation** (used in Table 6 and Section 5.4): 144
  triples, held out from the 1,446 preference dataset. Used to compute
  reward accuracy and reward margin during DPO training.
- **SFT validation** (used in Section 6.2 perplexity, Table 7): 750
  examples = 250 held-out records times three queries. Used to compute
  perplexity on the supervised fine-tuning distribution.

Verified: both are real, distinct sets. The report does describe each
correctly in its own section, but the distinction is not always made
explicit at the point of use. A reader skimming Table 6 then jumping to
Table 7 might try to mathematically reconcile the two.

**Recommended fix**: add one parenthetical clarification in Section 6.2
opening sentence: "Perplexity is computed on the SFT validation set (the
250 held-out records from Section 2.2, three queries each, 750 examples;
not the 144-triple DPO preference validation set used in Section 5.4)".

### E5. "Quantify the close-pair noise floor: effective N = 3-8" - **Valid addition**

The reviewer points out that close-pair AB-BA agreement of 15-40% on N=20
prompts corresponds to an effective sample size of just 3 to 8 resolved
matchups per pair. Verified arithmetic: 0.15 × 20 = 3 and 0.40 × 20 = 8.

This number sharpens the §7.4 "small consistent-verdict counts" argument
considerably: a 5-1 close-pair result is drawn from a sample of 6, not
20. The binomial-CI honesty in §7.4 implies this but does not state the
effective-N directly.

**Recommended addition (in §7.4 paragraph on N=20 limits)**:
> "The effective sample size per close-pair comparison is further
> reduced by the inconsistency rate: at 15-40% AB-BA agreement on N=20
> prompts, the binding sample is only 3 to 8 resolved matchups. The 8-0
> SFT-vs-DPO-b01 result is at the upper end of this range; the 3-0
> b01-vs-b02 result is at the lower end and is suggestive rather than
> conclusive on those grounds."

### E6. "Foreshadow the rep_penalty regression earlier" - **Valid presentation suggestion**

The reviewer suggests mentioning the `repetition_penalty=1.3` requirement
earlier in the report (Section 1 or 3.4) so a reader who stops before
Section 7 does not infer that the model is inherently unstable.

Currently:
- §3.4 sanity check: shows the SFT improvement using the helper that
  already carries `repetition_penalty=1.3` (cell `ma2s35code`) but does
  not explain the kwarg.
- §6.1: mentions `repetition_penalty=1.3` and forward-refs Section 7.
- §7.1: full diagnostic story.

The §6.1 forward-reference is already discoverable; a §1 or §3.4 mention
is optional polish.

**Recommended addition (optional)**: in §1 paragraph that lists the
methodological choices, add one sentence: "Inference-time decoder kwargs
turned out to be load-bearing on this model family; the
`repetition_penalty=1.3` setting carried forward from Mini-Assignment 1
is non-optional, and dropping it produces catastrophic failures
documented in §7.1."

### Updated action checklist (after second reviewer)

| Item | Source | Status | Priority |
|---|---|---|---|
| A1 | First-pass validation | Fix in LaTeX | high |
| A2 | First-pass validation | Fix in LaTeX | high |
| A3 | First-pass validation | Fix in notebook | high |
| B1 | First-pass tightening | Optional | low |
| B2 | First-pass tightening | Optional | low |
| E1 (DPO-6005 typos) | Second reviewer | **Not a real issue** | none |
| E2 (Table 4 blanks) | Second reviewer | Fix in LaTeX | medium |
| E3 (SFT loss masking) | Second reviewer | Optional §7 expansion | low |
| E4 (validation set nomenclature) | Second reviewer | Optional §6.2 clarification | low |
| E5 (effective-N math) | Second reviewer | Recommended §7.4 addition | medium |
| E6 (foreshadow rep_penalty) | Second reviewer | Optional §1 sentence | low |

---

## F. Validation of the third LLM reviewer's feedback

The third reviewer offers a substantive critique focused on hedging the
strength of conclusions and improving statistical treatment. Most points
are valid and align with weaknesses already identified, sometimes more
forcefully than my own first pass.

### F1. "Soften the b01 'alignment-quality peak' confidence" - **valid, overlaps with A2**

The reviewer asks the conclusion to soften:
> "The alignment-quality peak sits at β = 0.10"

to something like:
> "The available evidence suggests β = 0.10 is the strongest candidate among the tested configurations, although the low close-pair agreement rates prevent a statistically robust ranking between the in-range DPO variants."

This is consistent with A2 (overclaimed "no structural regression"): both
findings push for hedged language. The §6.5 Synthesis, §7.2 closing, and
the Conclusion (§8) all carry the same overconfident peak-locating
phrasing. The hedge belongs in all three.

**Recommended fix**: replace the categorical "peak sits at β=0.10" with
"strongest candidate among the tested configurations" or "best-performing
in this evaluation" wherever the categorical version appears. Three
specific instances to soften:
- §6.5 final paragraph ("The deliverable policy is DPO-b01...").
- §7.2 closing ("The alignment-quality peak sits at β = 0.10").
- §8 Conclusion ("located the alignment-quality peak at β = 0.10").

### F2. "Statistical significance mostly absent - add binomial CIs" - **valid, extends E5**

The reviewer notes that 8-0, 3-0, 5-1, 2-1 verdicts are reported as raw
counts without confidence intervals. A binomial 95% confidence interval on
the close-pair win-rates would sharpen the discussion:

| count | implied p | 95% CI on p (Wilson) |
|---|---|---|
| 8 of 8 | 1.00 | [0.68, 1.00] |
| 3 of 3 | 1.00 | [0.44, 1.00] |
| 5 of 6 | 0.83 | [0.44, 0.97] |
| 2 of 3 | 0.67 | [0.21, 0.94] |

These intervals translate the existing §7.4 "wide binomial CIs" gesture
into concrete numbers. The exercise also confirms that the 8-0 sweep is
robust at the 95% level (lower bound 68% well above 50% chance) while the
3-0 close-pair count is consistent with anything from 44% to 100% true
win-rate, which is genuinely suggestive rather than conclusive.

**Recommended addition**: a small two-row table in §7.4 ("Effective sample
size and 95% CI on observed win-rates for close-pair verdicts") covering
each of the close-pair counts reported in the body.

### F3. "Perplexity interpreted more strongly than warranted" - **valid clarification**

The reviewer points out that perplexity on the SFT validation set
*intentionally* rises under DPO (DPO moves the policy off the SFT
distribution by design), so a rise is not by itself evidence of worse
deployment quality. The cliff between b01 (6.01) and b005 (8.84) is
interpreted in §6.2 as "alignment gain no longer compensates for SFT
divergence", which is defensible, but the underlying point that
perplexity-on-SFT measures distance-from-SFT rather than absolute quality
deserves to be stated explicitly.

**Recommended addition** (one sentence after the Table 7 caption in §6.2):
> "Perplexity here measures distance from the SFT distribution rather
> than absolute output quality: a DPO policy is intentionally moving off
> SFT, so a moderate rise is expected. The metric becomes a quality
> signal only when the rise is disproportionate to the alignment gain,
> which is the threshold the b01-to-b005 jump appears to cross."

### F4. "N=20 too small for central claim" - **valid, overlaps with E5 and F2**

The reviewer points out that the practical consequence of N=20 is that "a
difference of only a few consistent verdicts can change the ranking between
DPO variants". The current §7.4 acknowledges this in general terms; the
reviewer wants the impact emphasised. Largely subsumed by F2's recommended
CI table, which makes the consequence concrete.

### F5. "Full-ranking 6 pairs are not independent observations" - **valid methodological caveat**

The reviewer notes that the six (higher, lower) pairs extracted from a
single four-candidate ranking (A>B, A>C, A>D, B>C, B>D, C>D) share
information: the candidate identities are reused across pairs, and the
ranking itself comes from one judge call on one prompt. Treating these as
1,446 independent triples (versus 222 best-worst triples in earlier
configurations) overstates the effective signal increase.

This connects strongly to the §7.4 "preference-data volume was not the
bottleneck" finding: the 6.5x scaling did not produce a 6.5x lift on any
deployment metric, which is exactly what one would predict if the pairs
are correlated. The non-independence is therefore not just a methodology
note but a possible explanation for the volume-was-not-the-lever result.

**Recommended addition**: one sentence in §4.4 (after "yielding 1,446
preference triples"):
> "The six pairs from one ranking are not statistically independent
> observations: the four candidate texts are reused and the underlying
> judge call is one. The effective signal increase over Configurations B
> and C is therefore smaller than the 6.5x factor suggests, which is
> consistent with the §7.4 finding that scaling preference volume did
> not produce a proportionate deployment-quality lift."

### F6. "Judge calibration methodology vulnerable - hedge earlier" - **valid presentation choice**

The reviewer suggests moving some of the §7.2 self-criticism (the 10-call
probe is statistically incapable of distinguishing 80% from 50% at any
useful confidence) back into §4.3 where the calibration is introduced. As
written, §4.3 reads as a defensible selection; §7.2 reads as if the
problem was discovered later. The honest framing is that the probe was
undersized by design (budget-driven choice) and the production-scale
ceiling exposed the cost.

**Recommended addition**: a hedging sentence at the end of §4.3, before
the "Nemotron is assigned the RLAIF role" paragraph:
> "The 10-call probe is a small sample by design (production scale was
> still to come at the calibration phase) and cannot rule out
> close-pair discrimination problems that production-scale evaluation
> later exposes. Section 7.2 develops this in full."

### F7. "Tone occasionally argumentative" - **partial valid, partial style choice**

The reviewer flags three terse phrasings as too informal for academic
prose:
> "The reward margin row is not a finding"
> "The arithmetic is informative."
> "Any Gemma prior compounded three times."

The first two are the user-preferred plain-language style documented in
[`agent-server-presets`](file:///home/logus/.claude/projects/-home-logus-env-iscte-atlm-pro/memory/report-writing-style.md). The
third ("Any Gemma prior compounded three times") is descriptive about
shared-prior contamination and is technically accurate.

Whether to soften is a presentation preference rather than a defect. If
the user prefers to match more conventional academic register on those
specific sentences, the reviewer's suggested rewordings are reasonable:
- "Reward margin is reported for completeness but should not be
  interpreted as a cross-beta quality metric." (for "not a finding")
- "The arithmetic clarifies the position-bias mechanism." (for "is
  informative")
- "Any stylistic preference Gemma encoded propagated through all three
  pipeline roles." (for "compounded three times")

**Recommended action**: user preference call.

### F8. "Conclusions stronger than evidence" - **overlaps with F1, A1**

The reviewer's example ("DPO wins every consistent verdict across all
four betas") is the exact A1 sentence already flagged as wrong (it
ignores the 2 SFT wins on the b005 leg). The reviewer's softening
("Among prompts where the judge reached a consistent verdict, DPO was
preferred in all observed SFT-vs-DPO comparisons") is also wrong for the
same reason. A1's recommended replacement is the correct framing:
restrict the universal claim to the three in-range betas.

### F9. "No visualizations - add three figures" - **valid presentation improvement**

The reviewer suggests three figures:
- **Figure A**: beta versus perplexity and reward accuracy (would show
  the b01 peak directly).
- **Figure B**: judge agreement bars by comparison class (Base-vs-aligned,
  SFT-vs-DPO, DPO-vs-DPO) (would visually expose the discrimination
  ceiling).
- **Figure C**: pipeline overview flow diagram (replaces or supplements
  Table 1).

All three would improve scanability. Figures A and B are essentially
visualisations of data already in Tables 6, 7 and 8, so producing them
is mechanical. Figure C is a stylistic choice.

**Recommended action**: Figure A and Figure B are clear wins; Figure C
is optional. Whether the user has time to author them is the budget
question, not the content question.

### F10. "Report exact generation settings everywhere" - **partially already done**

The reviewer wants explicit `top_p`, `temperature`, `max_new_tokens` at
every generation stage. Verified in the report:
- §3.4 sanity check: "greedy decoding" stated, no explicit
  `repetition_penalty`, `max_new_tokens` value.
- §4.2 candidate sampling: "temperature 0.9, nucleus sampling (p=0.95)",
  "num_return_sequences=4", "maximum of 500 new tokens" - complete.
- §6.1 evaluation generation: "greedy decoding and
  `repetition_penalty=1.3`" - no explicit `max_new_tokens` value
  (the notebook uses 800).
- §5.5 sanity check (Section 5.5): not described in detail in the
  report, currently uses `repetition_penalty=1.3` and `max_new_tokens=400`
  per the notebook.

**Recommended addition**: a small "Generation settings used in this work"
table either as an appendix or as a parenthetical in §6.1, listing all
stages with the four kwargs (decode mode, temperature, top_p,
max_new_tokens) and rep_penalty.

### F11. "Explain why β=0.10 chosen over β=0.20" - **valid decision-rule clarification**

The reviewer points out that β=0.20 has lower perplexity (5.03 vs 6.01)
but β=0.10 wins on reward accuracy (0.715 vs 0.694). The report selects
b01 but does not articulate the decision rule.

**Recommended addition**: one sentence in §6.5 before naming the
deliverable:
> "Among the in-range betas, perplexity is lowest at β=0.30 and rises as
> β decreases (the expected alignment-fluency trade-off direction), while
> the win-rate signal sorts in the opposite direction (DPO-b01 wins more
> consistent verdicts against the higher betas). Selecting the deliverable
> prioritises the alignment metric (preference accuracy and pairwise
> win-rate) over absolute fit to the SFT distribution, provided perplexity
> remains within a proportionate range of SFT (which it does at β=0.10:
> 6.01 vs SFT's 4.54, a 32% rise)."

---

## G. Final consolidated action checklist (all three reviewers)

| Item | Description | Source | Priority |
|---|---|---|---|
| A1 | §6.3 wrong count "0 SFT wins in 25" - fix to "18 of 18 across three in-range betas" | First-pass + F8 | **high** |
| A2 | §6.5 "no structural regression" overclaim - soften with Table 9 numbers | First-pass | **high** |
| A3 | Notebook §6.6 and §7.1 markdown "all four sections every time" claim - fix to match Table 9 | First-pass | **high** |
| F1 | Soften "alignment-quality peak sits at β=0.10" in three places (§6.5, §7.2, §8) | Third reviewer | high |
| F2 | Add binomial 95% CI table in §7.4 for close-pair verdict counts | Third reviewer | medium |
| F3 | One-sentence clarification in §6.2 that perplexity-on-SFT measures divergence, not absolute quality | Third reviewer | medium |
| F5 | One-sentence caveat in §4.4 that the 6 pairs from one ranking are not independent | Third reviewer | medium |
| F6 | Hedging sentence at end of §4.3 about calibration sample size limitations | Third reviewer | medium |
| F11 | One sentence in §6.5 explaining the decision rule (alignment-priority over perplexity-priority) | Third reviewer | medium |
| E2 | Fill Table 4 ood-03 blank cells with checkmarks | Second reviewer | medium |
| E5 | §7.4 effective-N math (subsumed by F2 if F2 is applied) | Second reviewer | medium |
| F9 | Add Figure A (β vs perplexity & reward acc) and Figure B (agreement bars) | Third reviewer | medium |
| B1 | §7.1 "42 SQS as bullet items" geometric description | First-pass | low |
| B2 | §5.4 "more epochs caused saturate" phrasing | First-pass | low |
| E3 | §7 expansion on unmasked-prompt-span SFT loss | Second reviewer | low |
| E4 | §6.2 parenthetical on SFT-validation vs DPO-validation set | Second reviewer | low |
| E6 | One-sentence foreshadow of rep_penalty in §1 or §3.4 | Second reviewer | low |
| F7 | Argumentative tone softening (three specific sentences) | Third reviewer | preference call |
| F10 | "Generation settings" table or parenthetical with all kwargs per stage | Third reviewer | preference call |
| E1 | DPO-6005 typo claim | Second reviewer | **not real** |

The eight items at "high" and "medium" priority are the load-bearing
corrections. Everything else is polish, structure, or preference. A1, A2,
A3 are the only items with on-disk-verified factual errors; F1 is the
only universal-claim softening that the data and three reviewers all
push for.

