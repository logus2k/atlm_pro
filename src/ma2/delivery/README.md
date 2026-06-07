# Mini-Assignment 2: Alignment of SmolLM2-360M for IT Job Postings

Authors: António Cruz, Bruno Santos, Pedro Miranda.

## Contents

- `atlm_ma2_groupc.ipynb`: the notebook, with all cell outputs preserved. It
  reads as the report.
- `requirements.txt`: pinned Python package versions matching the run that
  produced the reported numbers.
- `src/`: the Mini-Assignment 1 inference helper and the LoRA merge script
  referenced by the notebook.
- `documents/development/`: the cross-judge calibration battery report, the
  development log, and the `agent_server` agent presets used for the RLAIF
  and eval judges.

## Prerequisites

- Python 3.12.
- An NVIDIA CUDA GPU with bf16 support for re-execution. The reported
  numbers were produced on an RTX 4090.
- `agent_server` on `http://localhost:7701` with the two judge presets
  installed, if you intend to re-execute Sections 4.3 or 6.4. Install
  instructions are in
  `documents/development/agent_server_setup/README.md`.

## Environment setup

```
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Reading versus re-running

The notebook is saved with all cell outputs, so it can be read as the
deliverable without re-executing anything. The metric tables, the
qualitative side-by-side, the win-rate matrix and the Section 7 methodology
discussion are all complete in the saved state.

Re-running end to end additionally requires the Mini-Assignment 1 merged
base at `outputs/mp1-360m/merged/`, the supervised fine-tuning corpus at
`data/processed/converted.jsonl`, and `agent_server` running locally with
the two judge presets installed.
