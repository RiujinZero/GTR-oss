"""Can a 20k four-layer Transformer memorize 100 fixed prefix examples?"""

import torch
import torch.nn.functional as F

from parallel_prefix_benchmark import PrefixTransformer980, evaluate, make_prefix_batch


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrefixTransformer980(20, 4, 80, n_layers=4, require_980=False).to(device)
    assert model.parameter_count == 20_726
    generator = torch.Generator(device=device).manual_seed(1234)
    observed, targets = make_prefix_batch(100, 7, device, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    print(f"device={device} parameters={model.parameter_count} fixed_examples=100")
    for step in range(1001):
        model.train()
        logits = model(observed, output_slots=7)
        loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            correct = logits.argmax(-1) == targets
            print(f"step={step:4d} loss={loss.item():.4f} "
                  f"train_acc={correct.float().mean().item():.3f} "
                  f"train_exact={correct.all(1).float().mean().item():.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        train_logits = model(observed, output_slots=7)
        train_correct = train_logits.argmax(-1) == targets
    heldout = evaluate(model, 7, device, seed=98765, batch_size=4096)
    print(f"final_train correct={train_correct.sum(1).float().mean().item():.2f}/7 "
          f"acc={train_correct.float().mean().item():.3f} "
          f"exact={train_correct.all(1).float().mean().item():.3f}")
    print(f"heldout correct={heldout['correct']:.2f}/7 acc={heldout['accuracy']:.3f} "
          f"exact={heldout['exact']:.3f} loss={heldout['loss']:.4f}")


if __name__ == "__main__":
    main()
