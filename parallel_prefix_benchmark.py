"""Simultaneous modular-prefix formation and length-extrapolation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gator.optimizers.procoim import (
    add_procoim_arguments,
    procoim_config_from_args,
)
from gator.optimizers.slot_loss import SlotLossResult, cross_entropy_slot_loss
from gator.training import ProcoimTrainingEngine
from gtr_a_experiments import GTRFixedSlots, fixed_code


class PrefixTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_width):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1, self.ff2 = nn.Linear(d_model, ff_width), nn.Linear(ff_width, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention_scale = nn.Parameter(torch.ones(()))
        self.ff_scale = nn.Parameter(torch.ones(()))

    def forward(self, h, attn_mask=None):
        attended, _ = self.attention(h, h, h, need_weights=False, attn_mask=attn_mask)
        h = self.norm1(h + self.attention_scale * attended)
        return self.norm2(h + self.ff_scale * self.ff2(F.gelu(self.ff1(h))))


class PrefixTransformer980(nn.Module):
    """One-shot noncausal Transformer with fixed positions and 980 parameters."""

    def __init__(self, d_model=6, n_heads=2, ff_width=48, n_layers=1, require_980=True):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(16, d_model)
        self.position_projection = nn.Linear(4, d_model, bias=False)
        self.slot_type = nn.Embedding(2, d_model)
        self.output_query_bias = nn.Parameter(torch.zeros(d_model))
        self.layers = nn.ModuleList(
            PrefixTransformerBlock(d_model, n_heads, ff_width) for _ in range(n_layers)
        )
        self.condition_scale = nn.Parameter(torch.ones(()))
        self.query_scale = nn.Parameter(torch.ones(()))
        self.lm_bias = nn.Parameter(torch.zeros(16))
        if require_980:
            assert sum(p.numel() for p in self.parameters()) == 980

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, observed, *, output_slots, temperature=1.0):
        del temperature
        batch, observed_length = observed.shape
        total_length = observed_length + output_slots
        condition = self.condition_scale * self.token_embedding(observed)
        mask = self.token_embedding.weight.mean(0) + self.output_query_bias
        queries = self.query_scale * mask.view(1, 1, self.d_model).expand(batch, output_slots, -1)
        h = torch.cat([condition, queries], dim=1)
        h = h + self.position_projection(fixed_code(total_length, 4, observed.device))[None]
        types = torch.cat([
            torch.zeros(observed_length, dtype=torch.long, device=observed.device),
            torch.ones(output_slots, dtype=torch.long, device=observed.device),
        ])
        h = h + self.slot_type(types)[None]
        for layer in self.layers:
            h = layer(h)
        return h[:, observed_length:] @ self.token_embedding.weight.t() + self.lm_bias


class PrefixTransformerCausal(nn.Module):
    """Autoregressive, causally-masked Transformer: trained via teacher
    forcing (the true previous output token predicts the next one) and
    evaluated via real token-by-token generation (its own prediction fed
    back in as the next input) -- unlike PrefixTransformer980, which
    predicts all output slots simultaneously in one non-causal pass, this
    decodes output positions one at a time, the way a standard causal LM
    does."""

    def __init__(self, d_model=14, n_heads=2, ff_width=52, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(16, d_model)
        self.position_projection = nn.Linear(4, d_model, bias=False)
        self.slot_type = nn.Embedding(2, d_model)
        self.start_token = nn.Parameter(torch.zeros(d_model))
        self.layers = nn.ModuleList(
            PrefixTransformerBlock(d_model, n_heads, ff_width) for _ in range(n_layers)
        )
        self.condition_scale = nn.Parameter(torch.ones(()))
        self.lm_bias = nn.Parameter(torch.zeros(16))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _hidden_states(self, observed, decode_tokens):
        # decode_tokens: (batch, k) tokens already known/generated BEFORE
        # the position(s) being predicted (ground truth while teacher
        # forcing, the model's own predictions while generating). k may be 0.
        batch, observed_length = observed.shape
        k = decode_tokens.shape[1]
        decode_length = k + 1  # +1 for the leading start-token position
        total_length = observed_length + decode_length
        condition = self.condition_scale * self.token_embedding(observed)
        start = self.start_token.view(1, 1, -1).expand(batch, 1, -1)
        decode = torch.cat([start, self.token_embedding(decode_tokens)], dim=1) if k > 0 else start
        h = torch.cat([condition, decode], dim=1)
        h = h + self.position_projection(fixed_code(total_length, 4, observed.device))[None]
        types = torch.cat([
            torch.zeros(observed_length, dtype=torch.long, device=observed.device),
            torch.ones(decode_length, dtype=torch.long, device=observed.device),
        ])
        h = h + self.slot_type(types)[None]
        mask = torch.triu(
            torch.full((total_length, total_length), float("-inf"), device=observed.device),
            diagonal=1,
        )
        for layer in self.layers:
            h = layer(h, attn_mask=mask)
        return h

    def forward(self, observed, targets):
        """Teacher-forced training forward: one vectorized pass returns
        logits at every decode position, each predicting targets[:, t]
        from the observed tokens plus the TRUE targets[:, :t]."""
        h = self._hidden_states(observed, targets[:, :-1])
        decode_h = h[:, observed.shape[1]:]
        return decode_h @ self.token_embedding.weight.t() + self.lm_bias

    @torch.no_grad()
    def generate(self, observed, output_slots):
        """Real autoregressive generation: at each step, feeds the model's
        own argmax prediction back in as the next input -- no access to
        ground truth. This is the actual test of whether the model can
        produce all `output_slots` positions correctly on its own, not
        just under teacher forcing."""
        batch = observed.shape[0]
        device = observed.device
        generated = torch.zeros(batch, 0, dtype=torch.long, device=device)
        for _ in range(output_slots):
            h = self._hidden_states(observed, generated)
            logits = h[:, -1] @ self.token_embedding.weight.t() + self.lm_bias
            next_token = logits.argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


def make_prefix_batch(
    batch_size,
    length,
    device,
    *,
    generator=None,
    sampling_device=None,
):
    # Inputs: [start, d1, ..., d_length]. Deltas are fresh random values 0..11.
    sampling_device = torch.device(sampling_device or device)
    start_value = torch.randint(
        0,
        12,
        (batch_size,),
        device=sampling_device,
        generator=generator,
    )
    deltas = torch.randint(
        0,
        12,
        (batch_size, length),
        device=sampling_device,
        generator=generator,
    )
    observed = torch.cat([(start_value + 4)[:, None], deltas], dim=1)
    targets = 4 + (start_value[:, None] + deltas.cumsum(dim=1)).remainder(12)
    return observed.to(device), targets.to(device)


def make_balanced_prefix_batch(
    batch_size,
    length,
    device,
    *,
    generator=None,
    sampling_device=None,
):
    """Generate valid prefix examples with exact per-slot target marginals.

    Random microbatches contain exploitable class-count noise: a forward-only
    search can repeatedly clear a small loss target by changing vocabulary
    priors without learning the prefix dependency. This stratified sampler
    makes every target token occur equally often at every output slot while
    retaining fresh random pairings between starts and successive targets.
    """

    if batch_size <= 0 or batch_size % 12 != 0:
        raise ValueError("balanced prefix batch_size must be a positive multiple of 12")
    if length <= 0:
        raise ValueError("balanced prefix length must be positive")
    sampling_device = torch.device(sampling_device or device)
    balanced = torch.arange(
        batch_size,
        device=sampling_device,
        dtype=torch.long,
    ).remainder(12)

    def shuffled_balanced():
        permutation = torch.randperm(
            batch_size,
            device=sampling_device,
            generator=generator,
        )
        return balanced.index_select(0, permutation)

    start_value = shuffled_balanced()
    target_values = torch.stack(
        [shuffled_balanced() for _ in range(length)],
        dim=1,
    )
    previous = torch.cat(
        (start_value[:, None], target_values[:, :-1]),
        dim=1,
    )
    deltas = (target_values - previous).remainder(12)
    observed = torch.cat(((start_value + 4)[:, None], deltas), dim=1)
    targets = target_values + 4
    return observed.to(device), targets.to(device)


def prefix_slot_loss_provider(
    model,
    batch,
    forward_kwargs,
) -> SlotLossResult:
    observed, targets = batch
    logits = model(observed, **forward_kwargs)
    return cross_entropy_slot_loss(logits, targets)


@torch.no_grad()
def evaluate(
    model,
    length,
    device,
    seed=12345,
    batch_size=4096,
    model_kwargs=None,
    sampling_device=None,
):
    sampling_device = torch.device(sampling_device or device)
    generator = torch.Generator(device=sampling_device).manual_seed(
        seed + length
    )
    observed, targets = make_prefix_batch(
        batch_size,
        length,
        device,
        generator=generator,
        sampling_device=sampling_device,
    )
    logits = model(
        observed,
        output_slots=length,
        **(model_kwargs or {}),
    )
    correct = logits.argmax(-1) == targets
    return {
        "length": length,
        "loss": F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1)).item(),
        "correct": correct.sum(1).float().mean().item(),
        "accuracy": correct.float().mean().item(),
        "exact": correct.all(1).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["gtra", "gtraw", "gtrar", "gtrarw", "transformer"], default="gtraw")
    parser.add_argument("--relays", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--train-length", type=int, default=7)
    parser.add_argument("--max-train-length", type=int, default=None)
    parser.add_argument("--curriculum-end", type=float, default=0.8)
    parser.add_argument("--eval-lengths", nargs="+", type=int, default=[7, 14])
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--optimizer", choices=("adamw", "procoim"), default="adamw"
    )
    parser.add_argument("--log-every", type=int, default=1000)
    add_procoim_arguments(parser)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        parser.error("--steps, --batch-size, and --log-every must be positive")
    procoim_config = None
    if args.optimizer == "procoim":
        if args.procoim_anchor_buffer_size != 0:
            parser.error(
                "--procoim-anchor-buffer-size is supported by "
                "gator_linear_prefix_train.py, not this legacy benchmark"
            )
        try:
            procoim_config = procoim_config_from_args(args)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (PrefixTransformer980() if args.variant == "transformer" else
             GTRFixedSlots(args.variant, args.relays, args.d_model)).to(device)
    optimizer = None
    procoim_engine = None
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=8e-3, weight_decay=1e-4
        )
    else:
        assert procoim_config is not None
        procoim_engine = ProcoimTrainingEngine(model, procoim_config)
    print(
        f"device={device} parameters={model.parameter_count} "
        f"variant={args.variant} optimizer={args.optimizer}"
    )
    max_train_length = args.max_train_length or args.train_length
    n_increments = max_train_length - args.train_length
    last_length = None
    training_step = 0
    episode = 0
    next_log_step = args.log_every
    while training_step < args.steps:
        if n_increments:
            stage_width = args.steps * args.curriculum_end / n_increments
            increment = min(n_increments, int(training_step / stage_width))
            current_length = args.train_length + increment
        else:
            current_length = args.train_length
        if current_length != last_length:
            print(
                f"curriculum step={training_step} length={current_length}",
                flush=True,
            )
            last_length = current_length
        observed, targets = make_prefix_batch(args.batch_size, current_length, device)
        temperature = max(0.35, 1.25 * (0.997**training_step))
        procoim_result = None
        if args.optimizer == "procoim":
            assert procoim_engine is not None
            procoim_result = procoim_engine.run_episode(
                (observed, targets),
                prefix_slot_loss_provider,
                forward_kwargs={
                    "output_slots": current_length,
                    "temperature": temperature,
                },
                example_id=episode,
                max_forward_calls=args.steps - training_step,
            )
            training_step = procoim_engine.step_number
            loss_value = procoim_result.end_aggregate_loss
        else:
            logits = model(
                observed,
                output_slots=current_length,
                temperature=temperature,
            )
            loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_step += 1
            loss_value = float(loss.detach())
        episode += 1
        if training_step >= next_log_step or training_step >= args.steps:
            current = evaluate(model, current_length, device, batch_size=1024)
            details = ""
            if procoim_result is not None:
                details = (
                    f" episode={episode} "
                    f"episode_forwards={procoim_result.total_forward_calls} "
                    f"accepted={procoim_result.accepted_updates} "
                    f"protected={procoim_result.protected_slots} "
                    f"target_hit={procoim_result.slot_target_hit_rate:.3f}"
                )
            print(
                f"step={training_step} L={current_length} "
                f"correct={current['correct']:.2f}/{current_length} "
                f"acc={current['accuracy']:.3f} exact={current['exact']:.3f} "
                f"train_loss={loss_value:.4f}{details}",
                flush=True,
            )
            next_log_step = (
                training_step // args.log_every + 1
            ) * args.log_every
            if args.checkpoint:
                payload = {
                    "step": training_step,
                    "episodes": episode,
                    "length": current_length,
                    "model": model.state_dict(),
                    "optimizer_name": args.optimizer,
                }
                if optimizer is not None:
                    payload["optimizer"] = optimizer.state_dict()
                if procoim_engine is not None:
                    payload["procoim"] = procoim_engine.state_dict()
                torch.save(payload, Path(args.checkpoint))
    for length in args.eval_lengths:
        result = evaluate(model, length, device)
        print(f"eval L={length} loss={result['loss']:.4f} correct={result['correct']:.2f}/{length} "
              f"acc={result['accuracy']:.3f} exact={result['exact']:.3f}")


if __name__ == "__main__":
    main()
