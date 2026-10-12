"""Exactly 980-parameter causal Transformer baseline for the GTR toy task."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from gtr_toy import make_rule_batch


class TinyTransformerLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 16
        self.d_model = 6
        self.max_len = 16

        self.token_embedding = nn.Embedding(16, 6)
        self.position_embedding = nn.Embedding(16, 6)
        self.attention = nn.MultiheadAttention(6, 2, batch_first=True)
        self.norm1 = nn.LayerNorm(6)
        self.ff1 = nn.Linear(6, 44)
        self.ff2 = nn.Linear(44, 6)
        self.norm2 = nn.LayerNorm(6)

        # Learned residual scales are useful parameters and make the count exact.
        self.attention_scale = nn.Parameter(torch.ones(()))
        self.ff_scale = nn.Parameter(torch.ones(()))
        self.lm_bias = nn.Parameter(torch.zeros(16))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        h = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attended, _ = self.attention(
            h, h, h, attn_mask=causal_mask, need_weights=False
        )
        h = self.norm1(h + self.attention_scale * attended)
        ff = self.ff2(F.gelu(self.ff1(h)))
        h = self.norm2(h + self.ff_scale * ff)
        return h @ self.token_embedding.weight.t() + self.lm_bias


def train_demo(*, steps: int = 1000, batch_size: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformerLM().to(device)
    assert model.parameter_count == 980, model.parameter_count
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)

    print(f"device={device}  trainable_parameters={model.parameter_count}")
    for step in range(steps + 1):
        model.train()
        inputs, targets = make_rule_batch(batch_size, device=device)
        logits = model(inputs)
        # Match GTR: position 0 predicts a random token and is excluded.
        loss = F.cross_entropy(
            logits[:, 1:].reshape(-1, 16), targets[:, 1:].reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                predictions = logits[:, 1:].argmax(dim=-1)
                accuracy = (predictions == targets[:, 1:]).float().mean().item()
            print(f"step={step:4d}  lm_loss={loss.item():.4f}  accuracy={accuracy:.3f}")

    model.eval()
    inputs, targets = make_rule_batch(1, device=device)
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=-1)
    print("\ninput: ", inputs[0].tolist())
    print("target:", targets[0].tolist())
    print("pred:  ", predictions[0].tolist())
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_demo(steps=args.steps, batch_size=args.batch_size, seed=args.seed)
