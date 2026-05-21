"""MP1 — continued pretraining of SmolLM2-135M on the job-postings domain.

Experiment knob:  --mode full  (full fine-tuning)  vs  --mode lora  (LoRA).
Everything else is held constant via configs/train.yaml.

  .venv_atlm_pro/bin/python src/train.py --mode full
  .venv_atlm_pro/bin/python src/train.py --mode lora

Outputs (per mode) to outputs/mp1-<mode>/: the model/adapter, log_history.json
(loss curves) and summary.json.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]


def tokenize_and_chunk(jsonl_path, tokenizer, block_size):
    """Tokenize raw text, mark doc boundaries with EOS, pack into fixed blocks."""
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    def tok(batch):
        out = tokenizer(batch["text"])
        for ids in out["input_ids"]:
            ids.append(tokenizer.eos_token_id)
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

    def group(batch):
        ids = list(itertools.chain.from_iterable(batch["input_ids"]))
        n = (len(ids) // block_size) * block_size
        blocks = [ids[i:i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": blocks,
                "attention_mask": [[1] * block_size for _ in blocks]}

    return ds.map(group, batched=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    set_seed(cfg["seed"])
    mcfg = cfg["modes"][args.mode]
    tcfg = cfg["training"]
    out_dir = ROOT / "outputs" / f"mp1-{args.mode}"

    print(f"=== MP1 continued pretraining — mode: {args.mode} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"])

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=mcfg["lora_r"], lora_alpha=mcfg["lora_alpha"],
            lora_dropout=mcfg["lora_dropout"],
            target_modules=mcfg["lora_target_modules"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"full fine-tuning — trainable params: {total:,} (100%)", flush=True)

    block = cfg["block_size"]
    train_ds = tokenize_and_chunk(ROOT / cfg["data"]["train"], tokenizer, block)
    val_ds = tokenize_and_chunk(ROOT / cfg["data"]["val"], tokenizer, block)
    print(f"train blocks: {len(train_ds):,} | val blocks: {len(val_ds):,} "
          f"| block size: {block}", flush=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(mcfg["learning_rate"]),
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        bf16=tcfg["bf16"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        logging_steps=tcfg["logging_steps"],
        save_strategy="no",
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    final = trainer.evaluate()
    ppl = math.exp(final["eval_loss"])
    summary = {
        "mode": args.mode,
        "learning_rate": float(mcfg["learning_rate"]),
        "epochs": tcfg["epochs"],
        "train_blocks": len(train_ds),
        "minutes": round(minutes, 1),
        "final_val_loss": round(final["eval_loss"], 4),
        "final_val_perplexity": round(ppl, 2),
    }
    (out_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n=== {args.mode} done — {minutes:.1f} min ===", flush=True)
    print(f"  final val loss: {final['eval_loss']:.4f} | "
          f"val perplexity: {ppl:.2f}", flush=True)
    print(f"  saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
