"""Standalone smoke test for NVIDIA Transformer Engine on this rig.
Not part of the package -- throwaway verification script, WSL-only.
Tests exactly ONE recipe per process invocation (pass recipe name as argv[1])
so there is no shared te.Linear module state or output-interleaving ambiguity
between recipes. Checks: import, GPU visibility, real forward+backward.
"""
import sys

import torch

recipe_name = sys.argv[1] if len(sys.argv) > 1 else "bf16"

print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0), flush=True)

import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe

print("transformer_engine:", getattr(te, "__version__", "unknown"), flush=True)
print("testing recipe:", recipe_name, flush=True)

device = "cuda"
torch.manual_seed(0)

in_features, out_features, batch = 256, 512, 64
x = torch.randn(batch, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
linear = te.Linear(in_features, out_features, bias=True).to(device=device, dtype=torch.bfloat16)

if recipe_name == "bf16":
    out = linear(x)
    out.sum().backward()
    print("RESULT bf16: forward_finite=", torch.isfinite(out).all().item(),
          "grad_finite=", torch.isfinite(x.grad).all().item(), flush=True)
else:
    recipe_cls = getattr(te_recipe, recipe_name)
    recipe = recipe_cls()
    try:
        with te.autocast(enabled=True, recipe=recipe):
            out = linear(x)
        out.sum().backward()
        grad_is_none = x.grad is None
        finite_out = torch.isfinite(out).all().item()
        finite_grad = torch.isfinite(x.grad).all().item() if not grad_is_none else False
        print(f"RESULT {recipe_name}: forward_finite={finite_out} grad_finite={finite_grad} grad_is_none={grad_is_none}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT {recipe_name}: FAILED -> {type(exc).__name__}: {exc}", flush=True)

print("done", flush=True)
