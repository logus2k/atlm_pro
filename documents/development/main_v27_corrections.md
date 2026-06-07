# `main (27).pdf` - Corrections to apply

## 1) In Section 6.3 (page 10), where it says:

> "Second, on the SFT vs. DPO pairs, DPO wins every consistent verdict
> across all four betas (0 SFT wins in 25 consistent pair-prompt
> verdicts), with zero losses."

It should say:

> "Second, on the SFT vs. DPO pairs, DPO wins every consistent verdict on
> the three in-range betas (18 of 18 across the SFT-vs-DPO-b01, b02 and
> b03 pairs, zero losses). On the SFT vs. DPO-b005 pair the signal
> weakens (7 DPO wins, 2 SFT wins on consistent verdicts), the first
> hint of the β=0.05 cliff that perplexity flags more sharply in
> Section 6.2."

**Why**: Table 8 (next page) shows 2 SFT wins on the b005 leg, and the
"25 consistent verdicts" number conflates DPO-wins (25) with consistent
verdicts (27 = 25 DPO + 2 SFT). §6.5's "18 of 18 across the three
in-range betas" phrasing is correct; §6.3 should match it.

---

## 2) In Section 6.5 (page 12), where it says:

> "The deliverable policy is DPO-b01 (β = 0.10, learning rate 5×10⁻⁵): it
> beats the base 18-0 with 90% judge agreement, beats SFT 8-0 in
> consistent verdicts, and shows no fluency collapse or structural
> regression relative to SFT."

It should say:

> "The deliverable policy is DPO-b01 (β = 0.10, learning rate 5×10⁻⁵): it
> beats the base 18-0 with 90% judge agreement, beats SFT 8-0 in
> consistent verdicts, and shows no fluency collapse. Structural
> completeness is comparable to SFT (1/20 vs 3/20 all-four-section
> generations; 2.50 vs 2.80 average sections per generation): a small
> decrement that is a known cost of restoring `repetition_penalty=1.3`,
> which occasionally triggers an early EOS before the model reaches
> `## Requirements`. Section 7 documents this trade-off."

**Why**: Verified directly from `data/processed/ma2/eval_generations/*.jsonl`.
SFT has 3/20 all-four-section generations and 2.80 avg sections; b01 has
1/20 and 2.50 avg sections. The "no structural regression" claim
contradicts Table 9 on the same page.

---

## 3) In the notebook cell `ma2s66` (Section 6.6), where it says:

> "All four DPO models produce all four required Markdown headings on
> every one of the 20 evaluation generations."

It should say:

> "Structural completeness varies across models. SFT and DPO-b03 produce
> all four required Markdown headings on 3 of 20 generations; DPO-b01
> on 1 of 20; DPO-b02 and DPO-b005 on 0 of 20. Average required-heading
> counts per generation are 2.80 (SFT), 2.85 (b03), 2.50 (b01), 2.45
> (b02), 1.35 (b005). The shortfall is the rep_penalty trade-off:
> restoring `repetition_penalty=1.3` (Section 6.2 and Section 7.3.1)
> eliminates the token-loop catastrophes but occasionally triggers an
> early EOS before the model reaches `## Requirements`."

**Why**: Verified by direct count over
`data/processed/ma2/eval_generations/*.jsonl`. The notebook prose
contradicts the report's Table 9.

---

## 4) In the notebook cell `ma2s71` (Section 7.1), where it says:

> "all four DPO models produce the four required Markdown sections on
> every one of the 20 evaluation generations."

It should say:

> "structural completeness lands within a narrow band of SFT but is
> generally lower than the 4/20 all-four-section count one would want
> for production: 3/20 (SFT), 3/20 (b03), 1/20 (b01), 0/20 (b02), 0/20
> (b005). The shortfall is the known cost of the `repetition_penalty=1.3`
> kwarg restoration documented in §7.3.1."

**Why**: Same defect as #3, in the limitations chapter rather than the
behavioural-stats chapter. Both must be fixed together.

---

## 5) In Section 6.5 (page 12), where it says:

> "The β = 0.05 probe breaks this pattern on every axis: a 47% perplexity
> jump, a 40% drop in mean output length, and structural statistics
> worse than SFT, combined with a win-rate signal that becomes noise
> (DPO-b005 vs. DPO-b01 is 2-1 with 85% inconsistency). The
> alignment-quality peak sits at β = 0.10."

It should say:

> "The β = 0.05 probe breaks this pattern on every axis: a 47% perplexity
> jump, a 40% drop in mean output length, and structural statistics
> worse than SFT, combined with a win-rate signal at the noise floor
> (DPO-b005 vs. DPO-b01 is 2-1 with 85% inconsistency). Among the
> tested configurations, β = 0.10 is the strongest candidate by every
> measurable axis (eval reward accuracy peak at 0.715; consistent
> close-pair win-rate; proportionate perplexity rise); the low
> close-pair agreement rates documented in Section 7 prevent a
> statistically robust ranking between the three in-range DPO variants."

**Why**: The same close-pair agreement rates (15-40%) that the report
acknowledges in §7 prevent strict ordering between b01, b02 and b03 at
N=20. The categorical "peak sits at β = 0.10" overstates what the data
can support.

---

## 6) In Section 7.2 (page 13), the closing paragraph that asserts the
β = 0.10 peak. Where it says (search for the categorical
"alignment-quality peak" phrasing):

> "...the alignment-quality peak sits at β = 0.10..."

It should say:

> "...β = 0.10 is the strongest candidate among the tested configurations
> on every measurable axis, although the close-pair AB-BA agreement
> rates documented above prevent a statistically robust ranking between
> the three in-range DPO variants..."

**Why**: Same softening as #5, second occurrence.

---

## 7) In Section 8 Conclusion (page 15), where it says:

> "A four-point beta sensitivity probe located the alignment-quality peak
> at β = 0.10: perplexity and win-rate both degrade below that value,
> with β = 0.05 producing a 47% perplexity cliff and a 40% drop in mean
> output length."

It should say:

> "A four-point beta sensitivity probe identifies β = 0.10 as the
> strongest candidate among the tested configurations: perplexity and
> win-rate both degrade below that value, with β = 0.05 producing a 47%
> perplexity cliff and a 40% drop in mean output length. The
> close-pair AB-BA agreement rates documented in Section 7.2 prevent a
> statistically robust ranking between the three in-range DPO variants;
> β = 0.10 is the recommended deliverable on best-available-evidence
> grounds rather than as a proven optimum."

**Why**: Same softening as #5, third occurrence (the headline phrasing
in the Conclusion).

---

## 8) In Table 4 (page 6), the `ood-03` column currently shows:

| Model | ood-03 |
|---|---|
| gemma-4 (control) | (blank) |
| nemotron | (blank) |
| granite-3.3 | (blank) |
| smollm3 | × |
| qwen3.5 | × |
| ministral | (blank) |

It should show:

| Model | ood-03 |
|---|---|
| gemma-4 (control) | ✓ |
| nemotron | ✓ |
| granite-3.3 | ✓ |
| smollm3 | × |
| qwen3.5 | × |
| ministral | ✓ |

**Why**: Verified via `pdftotext` extraction that the cells are
genuinely blank in the rendered PDF. The semantic ("blank = caught the
collapse") is inferable but ambiguous; a reader scanning quickly may
assume "blank" means "not tested" or "data unavailable". Explicit
checkmarks resolve the ambiguity. The cells that pass the disqualifier
should display ✓ rather than nothing.

---

## 9) In Section 6.5 (page 12), where the deliverable is named without
explaining the decision rule. Where it says:

> "The deliverable policy is DPO-b01 (β = 0.10, learning rate 5×10⁻⁵): it
> beats the base 18-0 with 90% judge agreement, beats SFT 8-0 in
> consistent verdicts, and shows no fluency collapse or structural
> regression relative to SFT."

(this paragraph is also targeted by item 2 above; the rule clarification
should be applied to the same paragraph after the structural-regression
correction is in place)

A new sentence should be inserted before "The deliverable policy is
DPO-b01..." to articulate the decision rule:

> "Among the in-range betas, perplexity is lowest at β = 0.30 (4.80) and
> rises monotonically as β decreases, while reward accuracy peaks at
> β = 0.10 (0.715) and consistent close-pair verdicts sort
> lower-beta-preferred. Selecting the deliverable prioritises the
> alignment metrics (preference accuracy and pairwise win-rate) over
> absolute fit to the SFT distribution, provided perplexity remains
> within a proportionate range of SFT (which it does at β = 0.10:
> 6.01 vs SFT's 4.54, a 32% rise)."

**Why**: Verified by full report text scan: the report names DPO-b01 as
the deliverable but never states why over β = 0.20, which has lower
perplexity (5.03 vs 6.01). The selection rule is implicit. A reader's
natural question is left unanswered. The added sentence makes the
decision rule explicit.

---

## 10) In Section 5.4 (page 8), where it says:

> "Five epochs over 1,302 triples was determined empirically: fewer
> epochs left reward margins near zero; more caused the train reward
> accuracy to saturate at 1.00 with no corresponding deployment-side
> benefit (see Section 7)."

It should say:

> "Five epochs over 1,302 triples was determined empirically: fewer
> epochs left reward margins near zero; at five epochs the training
> reward accuracy already saturates at 1.00 across all four betas, so
> additional epochs would not improve the training-side signal and
> risk further fluency degradation (see Section 7)."

**Why**: Verified from `outputs/ma2-360m-dpo-*/log_history.json`:
train_rwd_acc reaches 1.00 at 5 epochs across all four betas. The
"more caused saturate" comparative implies a 6+ epoch experiment was
run; none was. The saturation is already at the chosen setting.

---

## 11) In Section 6.1 (page 9), where it says:

> "All six models are evaluated with greedy decoding and
> repetition_penalty=1.3, matching the Mini-Assignment 1 inference
> helper."

It should say:

> "All six models are evaluated with greedy decoding,
> `repetition_penalty=1.3`, and `max_new_tokens=800`, matching the
> Mini-Assignment 1 inference helper for the first two kwargs. The
> Section 3.4 sanity-check generations use the same decoder
> configuration with `max_new_tokens=400`; the Section 4.2 candidate
> sampling uses `temperature=0.9`, `top_p=0.95`,
> `num_return_sequences=4` and `max_new_tokens=500`."

**Why**: Verified by full report text scan: `max_new_tokens` is not
stated for §3.4 sanity-check generations or §6.1 evaluation generation,
though both are present and important for reproducibility. The notebook
cells `ma2s35code` (Section 3.5 sanity check) use 400, `ma2s55code`
(Section 5.5 sanity check) use 400, `ma2s62code` (Section 6.2 eval
generation) use 800. The consolidated sentence above replaces the
existing one-liner with a complete generation-kwarg summary covering
the three stochastic/greedy decoding stages.

---

## 12) In Section 7.1 (page 12), where it says:

> "DPO-β=0.20 on the cloud-engineer prompt ind-07 emitted 42 consecutive
> repetitions of the token SQS as bullet items in the Required Skills
> section; DPO-β=0.30 on the same prompt emitted 29 consecutive ECS
> repetitions."

It should say:

> "DPO-β=0.20 on the cloud-engineer prompt ind-07 emitted a Required
> Skills bullet in which the AWS service token SQS repeated 171 times
> in a comma-separated list before EOS; DPO-β=0.30 on the same prompt
> emitted an escalating-bullet degeneration in which the ECS token
> grew across consecutive bullet items ('ECS Pods', 'ECS ECS Pods',
> 'ECS ECS ECS Pods', ...) for over 25 lines."

**Why**: The "42" and "29" were sliding-window n-gram repeat counts from
the original diagnostic scanner, not bullet-item counts. The b02 case
manifests as a comma-separated list within a single bullet (171 raw SQS
tokens); only the b03 case presents as escalating bullet items. The
qualitative finding is unchanged; the geometric description matches the
actual data.

---

## Summary

| # | Location | Type | Priority |
|---|---|---|---|
| 1 | Report §6.3 | Factual count error, contradicts Table 8 | **high** |
| 2 | Report §6.5 | Overclaim, contradicts Table 9 on same page | **high** |
| 3 | Notebook `ma2s66` | All-sections claim contradicts on-disk data | **high** |
| 4 | Notebook `ma2s71` | Same defect as #3 in the limitations chapter | **high** |
| 5 | Report §6.5 | "Peak sits at" overconfident given §7 caveats | medium |
| 6 | Report §7.2 | Same overconfidence in beta-sensitivity closing | medium |
| 7 | Report §8 | Same overconfidence in Conclusion | medium |
| 8 | Report Table 4 | ood-03 blank cells, fill with ✓ for passing models | medium |
| 9 | Report §6.5 | Missing decision rule (b01 over b02) | medium |
| 10 | Report §5.4 | "More epochs caused saturate" implies untested | medium-low |
| 11 | Report §6.1 | `max_new_tokens` missing for §3.4 and §6.1 generation | low |
| 12 | Report §7.1 | "42 SQS as bullet items" geometry off | low |

Items 1-4 are factual contradictions visible to any careful reader of
the report (or report + notebook); they are the only items that produce
an internal inconsistency on inspection. Items 5-7 are the same
overconfidence pattern in three locations and can be fixed together.
Items 8, 9 are visible content gaps (a blank table column, a missing
decision rule for the deliverable). Item 10 implies an experiment that
was not run. Item 11 is a reproducibility kwarg gap. Item 12 is a
descriptive nuance where the underlying finding is unchanged.
