"""Build the Q8 GGUF of DPO-b01 for agent_server consumption.

Pipeline (three idempotent stages):

  1. Merge the DPO-b01 LoRA into its base (the SFT-merged SmolLM2-360M).
     Output: outputs/ma3/ma2-360m-dpo-b01-merged/   (~720 MB, bf16 HF)

  2. Convert the merged HF model to GGUF F16 via llama.cpp's converter.
     Output: outputs/ma3/ma2-360m-dpo-b01-F16.gguf  (~720 MB, intermediate)

  3. Quantize F16 -> Q8_0 via llama-quantize.
     Output: outputs/ma3/ma2-360m-dpo-b01-Q8_0.gguf (~380 MB, deliverable)

Re-runs skip any stage whose output already exists. Pass --force to rebuild
all stages. Pass --keep-f16 to retain the intermediate F16 GGUF; by default
it is removed after the Q8 quantisation succeeds.

Stages 2 and 3 are run inside the official llama.cpp docker image
`ghcr.io/ggml-org/llama.cpp:full` (no llama.cpp install on the host required).
That image ships the conversion script and the quantize binary, which the
agent_server runtime image does not. Override the image and tool paths with
these env vars if needed:

  LLAMA_CPP_IMAGE      default: ghcr.io/ggml-org/llama.cpp:full
  CONVERT_SCRIPT_PATH  default: /app/convert_hf_to_gguf.py
  QUANTIZE_BIN_PATH    default: /app/llama-quantize

To discover the actual paths inside a different image, run:

  docker run --rm --entrypoint sh <image> -c \\
    "find / -name convert_hf_to_gguf.py 2>/dev/null; \\
     find / -name llama-quantize 2>/dev/null"

Run from the repo root:  python src/ma3/build_b01_gguf.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = ROOT / "outputs" / "ma2-360m-dpo-b01"
OUT_DIR = ROOT / "outputs" / "ma3"
MERGED_DIR = OUT_DIR / "ma2-360m-dpo-b01-merged"
F16_GGUF = OUT_DIR / "ma2-360m-dpo-b01-F16.gguf"
Q8_GGUF = OUT_DIR / "ma2-360m-dpo-b01-Q8_0.gguf"

DOCKER_IMAGE = os.environ.get(
    "LLAMA_CPP_IMAGE", "ghcr.io/ggml-org/llama.cpp:full"
)
CONVERT_SCRIPT = os.environ.get(
    "CONVERT_SCRIPT_PATH", "/app/convert_hf_to_gguf.py"
)
QUANTIZE_BIN = os.environ.get(
    "QUANTIZE_BIN_PATH", "/app/llama-quantize"
)


def _mb(path: Path) -> int:
    return path.stat().st_size // (1024 * 1024)


def step_merge(force: bool = False) -> None:
    """Stage 1: merge DPO-b01 LoRA into the SFT-merged base."""
    if MERGED_DIR.exists() and any(MERGED_DIR.iterdir()) and not force:
        print(f"[merge]   skip — {MERGED_DIR.name}/ already exists")
        return

    # Read the base path from the LoRA's adapter_config (project convention:
    # never hardcode the base; the adapter knows what it was trained on).
    adapter_cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    base_path = Path(adapter_cfg["base_model_name_or_path"])

    # Defer heavy imports until we actually need them.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge]   base : {base_path}")
    print(f"[merge]   lora : {LORA_DIR}")
    print(f"[merge]   out  : {MERGED_DIR}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(
        model, str(LORA_DIR), torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    (MERGED_DIR / "README.md").write_text(
        "# DPO-b01 LoRA merged into the SFT-merged SmolLM2-360M\n\n"
        "Provenance:\n"
        f"- Base: `{base_path}` (SmolLM2-360M + MA1 LoRA + MA2 SFT, merged)\n"
        f"- LoRA: `{LORA_DIR.relative_to(ROOT)}` (DPO at beta=0.10, MA2 deliverable)\n"
        "- Built by: `src/ma3/build_b01_gguf.py`\n\n"
        "Self-contained MA3 starting point for GGUF conversion and Q8 quantisation.\n"
    )
    print(f"[merge]   done in {time.time() - t0:.1f}s")


def _run_in_image(*cmd_inside: str) -> None:
    """Run a command inside the llama.cpp docker image with outputs/ma3 mounted at /work.

    Files written into /work are owned by the host user thanks to --user.
    """
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{OUT_DIR}:/work",
        "--entrypoint", cmd_inside[0],
        DOCKER_IMAGE,
        *cmd_inside[1:],
    ]
    print(f"[docker]  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _normalize_tokenizer_config() -> None:
    """Patch tokenizer_config.json so the older transformers inside the docker
    image can load it.

    Newer transformers saves `extra_special_tokens` as a list (often `[]`).
    The transformers version inside `ghcr.io/ggml-org/llama.cpp:full` expects
    a dict at that key and crashes with:
        AttributeError: 'list' object has no attribute 'keys'
    Both representations carry zero entries when empty, so converting `[]`
    to `{}` is a no-op at inference. Idempotent.
    """
    cfg_path = MERGED_DIR / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    val = cfg.get("extra_special_tokens")
    if isinstance(val, list):
        cfg["extra_special_tokens"] = {x: x for x in val} if val else {}
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[normalize] patched extra_special_tokens (list -> dict) in {cfg_path.name}")


def step_convert(force: bool = False) -> None:
    """Stage 2: HF merged -> GGUF F16."""
    if F16_GGUF.exists() and not force:
        print(f"[convert] skip — {F16_GGUF.name} already exists ({_mb(F16_GGUF)} MB)")
        return
    _normalize_tokenizer_config()
    print(f"[convert] HF -> GGUF F16")
    t0 = time.time()
    _run_in_image(
        "python3",
        CONVERT_SCRIPT,
        f"/work/{MERGED_DIR.name}",
        "--outfile", f"/work/{F16_GGUF.name}",
        "--outtype", "f16",
    )
    print(f"[convert] done in {time.time() - t0:.1f}s  ({_mb(F16_GGUF)} MB)")


def step_quantize(force: bool = False) -> None:
    """Stage 3: GGUF F16 -> GGUF Q8_0."""
    if Q8_GGUF.exists() and not force:
        print(f"[quantize] skip — {Q8_GGUF.name} already exists ({_mb(Q8_GGUF)} MB)")
        return
    print(f"[quantize] F16 -> Q8_0")
    t0 = time.time()
    _run_in_image(
        QUANTIZE_BIN,
        f"/work/{F16_GGUF.name}",
        f"/work/{Q8_GGUF.name}",
        "Q8_0",
    )
    print(f"[quantize] done in {time.time() - t0:.1f}s  ({_mb(Q8_GGUF)} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild all stages even if outputs already exist.",
    )
    ap.add_argument(
        "--keep-f16", action="store_true",
        help="Retain the intermediate F16 GGUF (default: remove after Q8 succeeds).",
    )
    args = ap.parse_args()

    if not LORA_DIR.exists():
        sys.exit(f"ERROR: DPO-b01 LoRA not found at {LORA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step_merge(force=args.force)
    step_convert(force=args.force)
    step_quantize(force=args.force)

    if not args.keep_f16 and F16_GGUF.exists():
        print(f"[cleanup] removing intermediate {F16_GGUF.name}")
        F16_GGUF.unlink()

    print()
    print(f"DONE: {Q8_GGUF}  ({_mb(Q8_GGUF)} MB)")
    print()
    print("Next: copy or symlink this GGUF into your agent_server ./data/models/")
    print("directory, then add the model entry to agent_config.json.")


if __name__ == "__main__":
    main()
