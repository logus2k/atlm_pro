# Mini-Assignment 1: completion plan

Date: 2026-05-22

This is the plan to bring Mini-Assignment 1 to full coverage of the
assignment brief. The deliverable is the notebook `src/atlm_mp1_v4.ipynb`, which
serves two roles at once: the reproducible code, and the source of the technical
report (an exporter converts it into a Word document).

## Current status

The notebook revision is complete. The notebook was rebuilt from a flat
15-section layout into the six CRISP-DM phases: Business Understanding, Data
Understanding, Data Preparation, Modeling, Evaluation, Deployment. All prose was
rewritten for clarity and to the project style rules, an interpretation cell was
added after every analysis, every code cell was commented, and a consistency
pass confirmed that all cross-references resolve and all code cells compile. The
notebook has 112 cells.

`sentence-transformers` and `umap-learn` were installed so the semantic
embedding analysis (Section 2.8) runs as a true semantic analysis rather than a
vocabulary-based fallback. `requirements.txt` and `requirements.lock.txt` were
updated.

The notebook has been run end to end (phase B), cleanly. The remaining work,
phase C, closes the gaps identified against the assignment brief.

## What the brief requires and the notebook already covers

- A pretrained open-source model: SmolLM2-360M, a decoder model well under 1B
  parameters.
- Continued pretraining on a chosen domain: IT job postings, about 22.7 MB of
  training text, well above the 1 to 10 MB the brief asks for.
- At least one training optimization explored with controlled runs: full
  fine-tuning versus LoRA. The 135M versus 360M comparison adds a second axis.
- Framework: HuggingFace Transformers Trainer, with no custom training loop.
- Dataset justification: the business case and sources (Sections 1.1 to 1.5),
  the data understanding (Section 2), and the cleaning and corpus build
  (Section 3).
- Results analysis: training and validation loss curves (5.1), generations
  before and after training (5.3 and 6.1), and perplexity (5.2 and 5.4).
- Reproducible code: the notebook plus the `src/` scripts, with a fixed random
  seed and exact versions pinned in `requirements.lock.txt`.

## Gaps the plan closes

1. There is no dedicated "Difficulties encountered" section. The brief requires
   it and weights it.
2. The hardware used and the compute budget are not stated.
3. The architecture justification (model size and family, capacity versus
   compute) is thin.
4. There is no explicit reproducibility statement.
5. `README.md` is stale; it still describes the 135M model.

## Phase B: the run

The user runs `atlm_mp1_v4.ipynb` end to end, with seed 42 and no forced
determinism. This retrains the 360M model (roughly 30 to 45 minutes), validates
that the revised notebook executes cleanly, and produces the final numbers and
fresh embedded outputs.

## Phase C: post-run revision for full brief coverage

Phase C runs after phase B, in this order.

### C0. Verify the run

Confirm the notebook ran end to end with no cell errors, that
`outputs/mp1-360m/{full,lora}/` and `eval.json` exist, and that every cell has
fresh output. Fix any failed cell before continuing. This is the gate for
everything else.

### C1. Reconcile numbers

Check every numeric claim in the markdown against the run. The Data
Understanding figures and the corpus figures are deterministic and should be
unchanged. The Section 5 tables compute themselves from the run artifacts. The
Section 5.6 discussion was written qualitatively; confirm each claim still holds
against the fresh perplexity numbers.

### C2. Rewrite the two Section 2.8 interpretation cells

Rewrite the 2.8.1 (projection) and 2.8.2 (nearest-neighbour) interpretation
cells in plain report language, describing the real semantic-embedding result.
Remove all "in this run", "fallback", "not installed" and conditional framing.

### C3. Add Section 5.7, "Difficulties encountered"

A new section, required and weighted by the brief. Drafted from the real project
history:

- LoRA hit CUDA out-of-memory at micro-batch 16 on this transformers and peft
  stack; resolved with micro-batch 4 and gradient accumulation 4, keeping the
  effective batch size equal.
- The full fine-tuning versus LoRA comparison could not hold the learning rate
  equal; each method keeps its conventional value, so the comparison carries a
  learning-rate confound, reported honestly.
- transformers 5.9 removed a `TrainingArguments` option that the first training
  script used, which had to be adjusted.
- The project began on SmolLM2-135M; weak, repetitive generations led to
  redoing the whole experiment at 360M. What we would do differently: fix the
  model size deliberately at the start.
- Running two model sizes caused output files to collide, which forced the
  per-run `outputs/<run>/` folder scheme.
- What surprised us: how little the domain adaptation transferred out of domain.

Placement: Section 5.7, after 5.6. Kept distinct from 6.2, which is about the
model's limitations rather than the process.

### C4. Expand Section 4.1

Add, in plain prose:

- Hardware: a single NVIDIA RTX 4090 (24 GB), on WSL2.
- Compute budget: three epochs per run, the per-run wall-clock time, and the
  total.
- Architecture justification: why the SmolLM2 family (open, modern small-model
  family, one tokenizer shared across sizes, which made the 135M-to-360M
  comparison friction-free) and why 360M as the main model (it fits the 4090 for
  full fine-tuning, and Section 5.5 shows it clearly beats 135M, at roughly
  double the compute). This closes the capacity-versus-compute point.
- Reproducibility: seed fixed at 42 for the corpus build, training and sampling;
  exact versions pinned in `requirements.lock.txt`; and a note that GPU training
  is not bit-deterministic, so figures vary slightly between runs.

### C5. Update README.md

Rewrite for the 360M model and v4: how to run the notebook end to end, the
`src/` scripts (`train.py`, `evaluate_mp1.py`, `generate_mp1.py`, each run-aware
through the `--run` argument), `requirements.txt` and `requirements.lock.txt`
for exact versions, and the `outputs/<run>/` layout. 135M is kept as the
comparison baseline, not the primary model.

### C6. Final consistency pass

Re-run the structural verification (heading tree including the new 5.7, all
cross-references resolve, no banned characters, every code cell compiles), and
re-read every markdown cell to catch any remaining self-referential or process
phrasing.

## Deferred until C0 to C6 are all done

These must not be started before phase C is complete.

### D1. Report-length trim

The notebook far exceeds the report's 10-page cap. Full requirement coverage
comes first; the selection down to 10 pages is a separate later step.

### D2. Base-model storage

Optionally move the HuggingFace base-model cache, currently the machine-wide
`~/.cache/huggingface/hub`, into a project-local `data/models` through the
`HF_HOME` environment variable. The trained `outputs/` folder stays where it is.
Trade-offs: this duplicates roughly 694 MB already present in the shared cache,
and `data/models` would have to be gitignored. To be discussed after C6.

## Coverage map

| Brief requirement | Addressed by |
|---|---|
| Difficulties encountered section | C3 |
| Hardware used | C4 |
| Compute budget justification | C4 |
| Architecture justification (capacity vs compute) | C4 |
| Reproducibility and random seeds | C4 |
| README.md | C5 |
| Numbers correct, honest reporting | C1, C2 |
| Working code, final check | C0, C6 |

## Evaluation criteria the plan targets

- Correct implementation (40%): validated by phase B and C0; sensible choices
  documented in C4.
- Quality of analysis (40%): the interpretation cells, the honest discussion in
  5.6, and the Difficulties section in C3.
- Clarity of report (20%): the CRISP-DM structure and the prose revision,
  confirmed by C6.
