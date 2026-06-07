# Mini-Assignment 2: Alignment of SmolLM2-360M for IT Job Postings

Authors: António Cruz, Bruno Santos, Pedro Miranda.

This README covers (1) reading the deliverable as-is and (2) re-executing the
full pipeline end to end against the reported numbers.

---

## Contents

- `atlm_ma2_groupc.ipynb`: the notebook, with all cell outputs preserved. It
  reads as the report.
- `requirements.txt`: pinned Python package versions matching the run.
- `src/generate_mp1.py`: the Mini-Assignment 1 inference helper referenced by
  the notebook (Section 7.3.1 carries the prior-art reference).
- `src/merge_ma1_lora.py`: standalone CPU script that merges the
  Mini-Assignment 1 LoRA adapter into the base; Section 2.1 re-implements
  the same merge inline.
- `documents/development/agent_server_setup/`: the two `agent_server` agent
  presets (RLAIF judge + eval judge) plus install instructions.
- `documents/development/llm_models_performance.md`: the cross-judge
  calibration battery report.
- `documents/development/llm_calibration/`: raw per-model probe JSONs
  underlying the calibration report.
- `documents/development/ma2_implementation_report.md`: the development log.
- `agent_server_install/`: the four `agent_server` distribution files
  (`docker-compose.yml`, `download_models.sh`, `agent_config.json`, and the
  upstream `README.md`) needed to set up the local LLM serving stack used
  by Sections 4.3 and 6.4. Bundled here so re-execution does not require
  fetching anything external for the serving stack itself.

---

## Reading the deliverable

The notebook is saved with all cell outputs, so it can be read as the report
without re-executing anything. Open `atlm_ma2_groupc.ipynb` in JupyterLab or
VS Code. The metric tables (Tables 3, 5, 6, 7, 8, 9), the qualitative
side-by-side, the win-rate matrix, and the full Section 7 methodology
discussion are all complete in the saved state.

If reading is all you intend, you can stop here. The remainder of this README
covers what is needed to re-execute the pipeline.

---

## Re-executing the pipeline: prerequisites

- **Hardware**: an NVIDIA CUDA GPU with bf16 support. The reported numbers
  were produced on a single 24 GB RTX 4090. Roughly 50 GB of disk for the
  Python environment, the local LLM models, the Mini-Assignment 1
  artefacts, and intermediate outputs.
- **Software**: Python 3.12; Docker Engine with Compose v2; the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so containers can use the GPU; `curl` and a sane shell.

Verify the GPU is visible to Docker:

```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 1. Python environment

From the delivery directory:

```
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

---

## Step 2. `agent_server` (the local LLM serving stack)

Sections 4.3 (RLAIF judging) and 6.4 (eval judging) of the notebook call
into `agent_server`, a self-hosted OpenAI-compatible LLM serving stack that
loads GGUF models on the GPU via `llama.cpp`. Two Docker images publish the
stack:

| Image | Role |
|---|---|
| [`logus2k/agent-server`](https://hub.docker.com/r/logus2k/agent-server) | OpenAI-compatible API + admin dashboard on port **7701** |
| [`logus2k/agent-server-llama-adapter`](https://hub.docker.com/r/logus2k/agent-server-llama-adapter) | the `llama.cpp` server that serves the GGUFs |

Model weights are **not baked into the images** (tens of GB). They are
downloaded by a script the agent_server distribution provides.

### 2.1 The agent_server distribution

The four distribution files are bundled with this delivery at
`agent_server_install/`:

- `docker-compose.yml` - compose file for the two-image stack.
- `download_models.sh` - GGUF download script.
- `agent_config.json` - the model catalogue and per-model configuration.
- `README.md` - the upstream `agent_server` manual (reference for the
  admin dashboard, model-switching API, troubleshooting, and full model
  catalogue beyond the three chat models needed for Mini-Assignment 2).

Copy `agent_server_install/` to a fresh working directory outside the
delivery (the stack will create a `data/` subfolder and download GGUFs
that should not live inside the delivery zip), then `cd` into it for the
remaining steps in this section:

```
cp -r agent_server_install /path/to/agent_server_install_workdir
cd /path/to/agent_server_install_workdir
```

### 2.2 Download the chat models needed for Mini-Assignment 2

The notebook requires three chat models from the agent_server catalogue: the
default `gemma-4` (always resident), `nemotron` (for §4.3 RLAIF judging),
and `granite-3.3` (for §6.4 eval judging). Two embedder models are always
resident as well.

```
chmod +x download_models.sh
./download_models.sh                        # downloads everything (~30 GB)
```

Or, if disk is tight and you only need the three chat models actually
called by the Mini-Assignment 2 notebook plus the always-resident
embedders, download these GGUFs manually into `./data/models/`:

| File | Size | URL |
|---|---|---|
| `gemma-4-E4B-it-UD-Q4_K_XL.gguf` | 4.8 GB | https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf |
| `mmproj-F16.gguf` (Gemma vision adapter, required) | 0.9 GB | https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf |
| `chat_template_gemma-4.jinja` | tiny | https://huggingface.co/google/gemma-4-E4B-it/resolve/main/chat_template.jinja |
| `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | 2.6 GB | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf |
| `granite-3.3-2b-instruct-UD-Q8_K_XL.gguf` | 2.9 GB | https://huggingface.co/unsloth/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-UD-Q8_K_XL.gguf |
| `bge-m3-q8/bge-m3-Q8_0.gguf` | 0.6 GB | https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf |
| `bge-reranker-v2-m3-q8/bge-reranker-v2-m3-Q8_0.gguf` | 0.6 GB | https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf |

The two embedder files live in subfolders (`bge-m3-q8/` and
`bge-reranker-v2-m3-q8/`) inside `data/models/`. The Gemma chat template is
saved as `chat_template_gemma-4.jinja` alongside the GGUFs.

### 2.3 Boot the stack

```
mkdir -p data && cp agent_config.json data/agent_config.json
docker compose pull
docker compose up -d
docker compose logs -f                      # wait for llama-vision to load ~30-60 s
```

Verify the server is up:

```
curl http://localhost:7701/v1/models
curl http://localhost:7701/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4","messages":[{"role":"user","content":"hello"}]}'
```

The response from `/v1/models` should include `gemma-4`, `nemotron`,
`granite-3.3` and the embedders. The dashboard is at
http://localhost:7701/admin/.

`agent_server` keeps **one chat model resident at a time** to bound VRAM use.
The notebook handles the switch programmatically through a small admin-API
call defined in cell `ma2s40helpers`. From a cold start gemma-4 is the active
chat model; the cell switches to `nemotron` before §4.3 and to `granite-3.3`
before §6.4.

### 2.4 Install the two judge presets

With `agent_server` running, install the Mini-Assignment 2 agent presets
(`atlm_rlaif_judge` and `atlm_eval_judge`) following the instructions in
`documents/development/agent_server_setup/README.md`. This is one admin-API
POST per preset (a few seconds total) and persists the rubrics server-side.

---

## Step 3. Mini-Assignment 1 artefacts

The notebook expects two upstream artefacts on disk before it can run:

- The Mini-Assignment 1 merged base at `outputs/mp1-360m/merged/` (a 694 MB
  consolidated checkpoint). The Mini-Assignment 1 delivery bundle contains
  the LoRA from which this base is built; `src/merge_ma1_lora.py` produces
  the merged checkpoint on CPU in a few seconds. Section 2.1 of the notebook
  also re-implements the merge inline.
- The supervised fine-tuning corpus at `data/processed/converted.jsonl`
  (2,507 records, produced by the Gemma-4 ETL teacher over raw Djinni
  postings). The notebook reads this in Section 3.2.

These two paths are relative to the directory in which the notebook is
launched.

---

## Step 4. Run the notebook

Open `atlm_ma2_groupc.ipynb` in JupyterLab or VS Code, point the kernel at
`.venv`, and run cells in order. Expected wall-clock on the reference
hardware (single RTX 4090):

| Stage | Approx. time |
|---|---|
| Section 2 (Mini-Assignment 1 merge) | < 1 min CPU |
| Section 3.4 (SFT training) | ~15 min GPU |
| Section 4.2 (candidate sampling) | ~30-45 min GPU |
| Section 4.3 (RLAIF judging via Nemotron) | ~15 min |
| Section 4.4 (assembly) | seconds, CPU |
| Section 5.4 (four DPO runs back to back) | ~2 hours GPU |
| Section 6.2 (six-way evaluation generation) | ~15 min GPU |
| Section 6.3 (perplexity) | ~5 min GPU |
| Section 6.4 (Granite win-rate) | ~10 min |
| Section 6.5 / 6.6 / 6.7 (display + synthesis) | instant, CPU |

Total end-to-end: roughly four hours from a cold cache.

The cells that hit `agent_server` (§4.3 and §6.4) carry idempotent resume
logic: if interrupted, just re-run the cell and it picks up where it left
off. The DPO training cell (`ma2s54run` in §5.4) is idempotent on
`summary.json` existence; runs that already completed are skipped on
re-execution.
