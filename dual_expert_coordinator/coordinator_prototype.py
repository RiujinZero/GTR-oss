import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/content/GTR_coordinator")

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from gator_convergent import ConvergentGator

_ACCEPTED_KEYS = set(inspect.signature(ConvergentGator.__init__).parameters.keys()) - {"self"}

MATH_CKPT = Path("/content/expert_math_final.pt")
LANG_CKPT = Path("/content/expert_language_final.pt")
MATH_DATA = Path("/content/modular_prefix_clean_train.jsonl")
LANG_DATA = Path("/content/smoltalk_preference_valid.jsonl")
HELDOUT_MATH = Path("/content/modmath_heldout_probe.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_expert(ckpt_path, config_overrides=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    filtered_config = {k: v for k, v in model_config.items() if k in _ACCEPTED_KEYS}
    if config_overrides:
        filtered_config.update(config_overrides)
    model = ConvergentGator(**filtered_config).to(device)
    state_dict = ckpt["model"]
    if any(k.startswith("core.") for k in state_dict):
        state_dict = {k[len("core."):]: v for k, v in state_dict.items() if k.startswith("core.")}
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model_config


print("loading math expert...", flush=True)
math_model, math_config = load_expert(MATH_CKPT)
print("loading language expert...", flush=True)
lang_model, lang_config = load_expert(LANG_CKPT, config_overrides={"vocabulary_mode": "shared_projected"})
print(f"math_config keys: {sorted(math_config.keys())[:10]}...", flush=True)
print(f"math params: {sum(p.numel() for p in math_model.parameters())}", flush=True)
print(f"lang params: {sum(p.numel() for p in lang_model.parameters())}", flush=True)

# ---- Router: tiny char-histogram MLP classifier (math=1, language=0) ----
CHARSET = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"")
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
VOCAB_SIZE = len(CHARSET) + 1  # +1 for unknown


def featurize(text: str) -> torch.Tensor:
    hist = torch.zeros(VOCAB_SIZE)
    for ch in text[:200]:
        idx = CHAR_TO_IDX.get(ch, VOCAB_SIZE - 1)
        hist[idx] += 1.0
    total = hist.sum().clamp(min=1.0)
    return hist / total


class Router(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_prompts(path, limit):
    prompts = []
    with open(path) as f:
        for line in f:
            if len(prompts) >= limit:
                break
            row = json.loads(line)
            text = row.get("text", "")
            if "USER:" in text:
                user_part = text.split("USER:", 1)[1]
                user_part = user_part.split("ASSISTANT:", 1)[0].strip()
                if user_part:
                    prompts.append(user_part)
    return prompts


print("loading router training data...", flush=True)
math_prompts = load_prompts(MATH_DATA, 4000)
lang_prompts = load_prompts(LANG_DATA, 4000)
print(f"math_prompts={len(math_prompts)} lang_prompts={len(lang_prompts)}", flush=True)

examples = [(p, 1) for p in math_prompts] + [(p, 0) for p in lang_prompts]
random.Random(0).shuffle(examples)
split = int(len(examples) * 0.9)
train_examples, valid_examples = examples[:split], examples[split:]

router = Router(VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(router.parameters(), lr=1e-2)

train_feats = torch.stack([featurize(p) for p, _ in train_examples]).to(device)
train_labels = torch.tensor([l for _, l in train_examples], dtype=torch.float32, device=device)
valid_feats = torch.stack([featurize(p) for p, _ in valid_examples]).to(device)
valid_labels = torch.tensor([l for _, l in valid_examples], dtype=torch.float32, device=device)

print("training router...", flush=True)
for epoch in range(200):
    router.train()
    optimizer.zero_grad()
    logits = router(train_feats)
    loss = F.binary_cross_entropy_with_logits(logits, train_labels)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0 or epoch == 199:
        router.eval()
        with torch.no_grad():
            val_logits = router(valid_feats)
            val_acc = ((val_logits > 0).float() == valid_labels).float().mean().item()
        print(f"epoch={epoch} train_loss={loss.item():.4f} val_acc={val_acc:.4f}", flush=True)

router.eval()

# ---- End-to-end dispatch test on held-out math probes + a language sample ----
def route_and_report(prompt: str):
    feat = featurize(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(router(feat)).item()
    chosen = "math" if score > 0.5 else "language"
    return chosen, score


print("\n--- dispatch spot-check ---", flush=True)
heldout_math_prompts = load_prompts(HELDOUT_MATH, 20)
for p in heldout_math_prompts[:10]:
    chosen, score = route_and_report(p)
    print(f"[math-heldout] prompt={p!r} routed={chosen} score={score:.3f}", flush=True)

for p in lang_prompts[:10]:
    chosen, score = route_and_report(p)
    print(f"[language] prompt={p!r} routed={chosen} score={score:.3f}", flush=True)

print("\nrouter parameter count:", sum(p.numel() for p in router.parameters()), flush=True)
torch.save(router.state_dict(), "/content/coordinator_router.pt")
print("saved router checkpoint", flush=True)
