import torch
from torch.backends.cuda import (
    flash_sdp_enabled, mem_efficient_sdp_enabled, math_sdp_enabled
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM

print("torch:", torch.__version__)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("flash flag:", flash_sdp_enabled(),
      "| mem-eff flag:", mem_efficient_sdp_enabled(),
      "| math flag:", math_sdp_enabled())

# Definitive: does each backend actually execute on this card?
device, dtype = "cuda", torch.bfloat16
q = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)
for backend, name in [
    (SDPBackend.FLASH_ATTENTION, "flash"),
    (SDPBackend.EFFICIENT_ATTENTION, "mem_efficient"),
    (SDPBackend.MATH, "math"),
]:
    try:
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAILED -> {type(e).__name__}: {str(e)[:120]}")

# What the actual model resolves to
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
print("attn_implementation:", model.config._attn_implementation)

