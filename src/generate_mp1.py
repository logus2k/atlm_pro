"""MP1 — interactive generation.

Feed the continued-pretrained model your own prompts and watch it complete a
job posting. MP1 is a *continued-pretraining* model — a text completer, not a
chat model: give it the START of a posting and it continues. Instruction-style
prompts ("write a JD for ...") are MP2's job.

  # one-shot
  .venv_atlm_pro/bin/python src/generate_mp1.py --model lora --prompt "We are looking for"

  # compare all three checkpoints on the same prompt
  .venv_atlm_pro/bin/python src/generate_mp1.py --model all --prompt "Senior Data Engineer"

  # REPL — type one prompt per line; a blank line or Ctrl-D quits
  .venv_atlm_pro/bin/python src/generate_mp1.py --model lora
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

CHECKPOINTS = {
    "base": BASE,                              # untrained reference
    "full": str(ROOT / "outputs" / "mp1-full"),  # full fine-tuning
    "lora": str(ROOT / "outputs" / "mp1-lora"),  # LoRA adapter (base + adapter)
}


def load_model(name):
    """Load 'base' | 'full' | 'lora' -> (model, tokenizer), eval mode on DEVICE."""
    if name not in CHECKPOINTS:
        raise ValueError(f"model must be one of {list(CHECKPOINTS)} (or 'all')")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if name == "lora":
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(BASE)
        model = PeftModel.from_pretrained(model, CHECKPOINTS["lora"])
    else:
        model = AutoModelForCausalLM.from_pretrained(CHECKPOINTS[name])
    return model.to(DEVICE, dtype=DTYPE).eval(), tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8,
             top_p=0.95, greedy=False, seed=42):
    """Complete `prompt`. Returns the full decoded string (prompt + continuation).

    Sampling (temperature / top-p) is the default; pass greedy=True for
    deterministic decoding. repetition_penalty curbs the loop-degeneration a
    135M model is prone to.
    """
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.3,
                  pad_token_id=tokenizer.eos_token_id)
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    out = model.generate(**enc, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description="Interactive MP1 text generation.")
    ap.add_argument("--model", default="lora",
                    choices=["base", "full", "lora", "all"],
                    help="checkpoint to use; 'all' compares the three")
    ap.add_argument("--prompt", default=None,
                    help="one-shot prompt; omit to enter the REPL")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true",
                    help="deterministic greedy decoding (ignores temperature/top-p)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = ["base", "full", "lora"] if args.model == "all" else [args.model]
    print(f"loading {names} on {DEVICE} ...", flush=True)
    loaded = {n: load_model(n) for n in names}
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature, top_p=args.top_p,
                      greedy=args.greedy, seed=args.seed)

    def run(prompt):
        for n in names:
            model, tokenizer = loaded[n]
            text = generate(model, tokenizer, prompt, **gen_kwargs)
            print(f"\n{'=' * 78}\n[{n}]  {prompt!r}\n{'-' * 78}\n{text}")
        print()

    if args.prompt is not None:
        run(args.prompt)
        return

    print("\nREPL — type the start of a job posting; blank line or Ctrl-D quits.")
    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        run(prompt)
    print("bye")


if __name__ == "__main__":
    main()
