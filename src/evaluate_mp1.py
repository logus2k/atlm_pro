"""MP1 — evaluate continued pretraining.

Perplexity for base vs full-FT vs LoRA, on in-domain (Djinni test) and
out-of-domain (LinkedIn) text, plus greedy sample generations from each model.

  .venv_atlm_pro/bin/python src/evaluate_mp1.py

Results -> outputs/mp1_eval.json
"""
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "HuggingFaceTB/SmolLM2-360M"
BLOCK = 1024
DEVICE = "cuda"
DTYPE = torch.bfloat16

MODELS = {
    "base": BASE,
    "full": str(ROOT / "outputs/mp1-full"),
    "lora": str(ROOT / "outputs/mp1-lora"),
}
EVAL_SETS = {
    "in_domain": ROOT / "data/processed/mp1/test.jsonl",
    "ood": ROOT / "data/processed/mp1/ood_test.jsonl",
}
PROMPTS = [
    "We are looking for a",
    "The ideal candidate will",
    "Responsibilities:\n-",
]


def make_blocks(jsonl, tokenizer):
    ds = load_dataset("json", data_files=str(jsonl), split="train")
    ids = []
    for text in ds["text"]:
        ids.extend(tokenizer(text)["input_ids"])
        ids.append(tokenizer.eos_token_id)
    n = (len(ids) // BLOCK) * BLOCK
    return [ids[i:i + BLOCK] for i in range(0, n, BLOCK)]


@torch.no_grad()
def perplexity(model, blocks):
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(blocks), 8):
        batch = torch.tensor(blocks[i:i + 8], device=DEVICE)
        loss = model(batch, labels=batch).loss
        n = batch.shape[0] * (batch.shape[1] - 1)
        tot_loss += loss.item() * n
        tot_tok += n
    return math.exp(tot_loss / tot_tok)


def load_model(name, spec):
    if name == "lora":
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(BASE)
        model = PeftModel.from_pretrained(model, spec)
    else:
        model = AutoModelForCausalLM.from_pretrained(spec)
    return model.to(DEVICE, dtype=DTYPE).eval()


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = {n: make_blocks(p, tokenizer) for n, p in EVAL_SETS.items()}
    for n, b in blocks.items():
        print(f"{n}: {len(b)} blocks")

    results = {"perplexity": {}, "generations": {}}
    for name, spec in MODELS.items():
        print(f"\n--- {name} ---", flush=True)
        model = load_model(name, spec)

        results["perplexity"][name] = {}
        for sname, b in blocks.items():
            ppl = perplexity(model, b)
            results["perplexity"][name][sname] = round(ppl, 2)
            print(f"  perplexity {sname:10}: {ppl:.2f}", flush=True)

        gens = []
        for prompt in PROMPTS:
            enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            gens.append({"prompt": prompt,
                         "output": tokenizer.decode(out[0], skip_special_tokens=True)})
        results["generations"][name] = gens

        del model
        torch.cuda.empty_cache()

    out_path = ROOT / "outputs/mp1_eval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== PERPLEXITY (lower = better) ===")
    print(f"  {'model':6} | {'in-domain':>10} | {'OOD':>10}")
    for m in MODELS:
        pp = results["perplexity"][m]
        print(f"  {m:6} | {pp['in_domain']:>10} | {pp['ood']:>10}")
    print(f"\nfull results (incl. generations) -> {out_path}")


if __name__ == "__main__":
    main()
