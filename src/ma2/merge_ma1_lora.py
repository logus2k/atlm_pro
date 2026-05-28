"""Merge the MA1 LoRA adapter into the SmolLM2-360M base.

Output: outputs/mp1-360m/merged/ (a self-contained model that is identical to
'base + MA1 LoRA' at inference). This is the starting point for the MA2
SFT and DPO stages.

Runs on CPU so it does not interfere with anything using the GPU.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide the GPU from this process

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = ROOT / "outputs" / "mp1-360m" / "lora"
OUT_DIR = ROOT / "outputs" / "mp1-360m" / "merged"

# Project convention: the base model id is read from the adapter, never hardcoded.
adapter_cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
BASE = adapter_cfg["base_model_name_or_path"]

print(f"Base model    : {BASE}")
print(f"LoRA adapter  : {LORA_DIR}")
print(f"Output (merged): {OUT_DIR}")
print()

t0 = time.time()
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE)

# Load base in bf16, matching the precision MA1 was trained in.
# device_map left unset so the model stays on CPU.
print("Loading base in bf16 (CPU) ...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)

print("Attaching LoRA adapter ...")
model = PeftModel.from_pretrained(model, str(LORA_DIR), torch_dtype=torch.bfloat16)

print("Merging LoRA weights into the base ...")
model = model.merge_and_unload()

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving merged model to {OUT_DIR} ...")
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# Provenance note alongside the merged model.
(OUT_DIR / "README.md").write_text(
    "# MA1 LoRA merged into SmolLM2-360M\n\n"
    "This folder is the MA1 LoRA adapter merged into the SmolLM2-360M base, in "
    "bf16. Inference from this folder is equivalent to loading the base model "
    "and the MA1 LoRA together, but with no PEFT plumbing.\n\n"
    "Provenance:\n"
    f"- Base model: `{BASE}`\n"
    f"- Source LoRA adapter: `outputs/mp1-360m/lora/`\n"
    "- Built by: `src/ma2/merge_ma1_lora.py`\n\n"
    "Purpose: starting point for the Mini-Assignment 2 SFT + DPO stages.\n"
)
print(f"Done in {time.time()-t0:.1f}s.")
