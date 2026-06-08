#!/usr/bin/env python3
"""Command-line tester for the MA2 DPO-b01 model.

Loads the SFT-merged base plus the DPO-b01 LoRA adapter, formats a recruiter
request through the MA2 SFT template (matching `ma2s35code` / `ma2s62code` in
the notebook), and prints the model's job-posting completion.

Inference settings match MA2 §11 final-pass defaults: bf16, repetition_penalty=1.3
(the kwarg whose absence caused the v6 first-pass catastrophe), pad_token_id
synced to EOS.

Usage:
    python src/ma2/test_b01.py "We need a Django backend engineer on AWS"
    python src/ma2/test_b01.py "..." --greedy
    python src/ma2/test_b01.py "..." --max-new-tokens 800 --seed 7
    python src/ma2/test_b01.py "..." --lora-dir outputs/ma2-360m-dpo-b02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "outputs" / "ma2-360m-sft-merged"
LORA_DIR_DEFAULT = ROOT / "outputs" / "ma2-360m-dpo-b01"

TEMPLATE = (
    "You are a recruitment assistant. Given a brief recruiter request, "
    "write a complete structured job posting in Markdown.\n\n"
    "### Request\n{query}\n\n### Posting\n"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="recruiter request")
    ap.add_argument("--lora-dir", default=str(LORA_DIR_DEFAULT),
                    help="path to LoRA adapter (default: ma2-360m-dpo-b01)")
    ap.add_argument("--sample", action="store_true",
                    help="stochastic decoding (default: greedy, matching notebook ma2s62code)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.lora_dir, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    prompt = TEMPLATE.format(query=args.query)
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)

    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
