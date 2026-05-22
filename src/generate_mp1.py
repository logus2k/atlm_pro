"""MP1 — interactive generation.

Feed a continued-pretrained model your own prompts and watch it complete a job
posting. MP1 is a *continued-pretraining* model — a text completer, not a chat
model: give it the START of a posting and it continues.

Each training run lives under `outputs/<run>/` (e.g. `outputs/mp1-360m/`),
holding `full/`, `lora/` and `eval.json`. Pass `--run` to pick which.

  # one-shot
  .venv_atlm_pro/bin/python src/generate_mp1.py --run mp1-360m --model lora --prompt "We are looking for"

  # compare base / full / lora on the same prompt
  .venv_atlm_pro/bin/python src/generate_mp1.py --run mp1-360m --model all --prompt "Senior Data Engineer"

  # REPL — type one prompt per line; a blank line or Ctrl-D quits
  .venv_atlm_pro/bin/python src/generate_mp1.py --run mp1-135m --model lora
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VALID = ("base", "full", "lora")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def base_model_id(run):
    """Base model id, read from the run's LoRA adapter config — never hardcoded,
    so it always matches whatever model that run was trained on."""
    cfg = json.loads((OUTPUTS / run / "lora" / "adapter_config.json").read_text())
    return cfg["base_model_name_or_path"]


def load_model(name, run):
    """Load 'base' | 'full' | 'lora' for a run -> (model, tokenizer), eval mode.

    `run` selects the output folder, e.g. 'mp1-360m' -> outputs/mp1-360m/."""
    if name not in VALID:
        raise ValueError(f"model must be one of {list(VALID)} (or 'all')")
    run_dir = OUTPUTS / run

    if name == "lora":
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(str(run_dir / "lora"))
        model = AutoModelForCausalLM.from_pretrained(base_model_id(run))
        model = PeftModel.from_pretrained(model, str(run_dir / "lora"))
    elif name == "full":
        tokenizer = AutoTokenizer.from_pretrained(str(run_dir / "full"))
        model = AutoModelForCausalLM.from_pretrained(str(run_dir / "full"))
    else:  # base
        base = base_model_id(run)
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(DEVICE, dtype=DTYPE).eval(), tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8,
             top_p=0.95, greedy=False, seed=42):
    """Complete `prompt`. Returns the full decoded string (prompt + continuation).

    Sampling (temperature / top-p) is the default; pass greedy=True for
    deterministic decoding. repetition_penalty curbs loop-degeneration.
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
    ap.add_argument("--run", default="mp1-360m",
                    help="output folder under outputs/ (e.g. mp1-135m, mp1-360m)")
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
    print(f"loading {names} from outputs/{args.run}/ on {DEVICE} ...", flush=True)
    loaded = {n: load_model(n, args.run) for n in names}
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
