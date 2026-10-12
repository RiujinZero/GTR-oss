# Graph Trace Relay (GTR / Gator)

Experimental code for Graph Trace Relay, a recurrent graph-routing architecture that refines and
emits a complete output sequence simultaneously instead of autoregressively.

> **Unmaintained.** This is a research prototype released as-is. It is no longer developed or
> supported; issues and pull requests are not monitored.

## What it is

Gator separates sequence position from relay depth. Every context token and output slot carries a
mutable state and a probabilistic location on a learned node graph. At each relay hop all positions
update in parallel: local node decisions route them, traffic that lands on the same node interacts
through graph memory, and after a fixed number of relays the whole output is read out at once.

The main track in this repository is **Convergent Gator**, in which one shared graph carries several
persistent payload lanes that are combined by a slot-conditioned terminal jury. Earlier variants
(the original 980-parameter toy, fixed-slot formation, Gator-Path, Gator-S stewardship,
Gator-Linear) are kept alongside it, together with matched Transformer controls.

## Layout

| Path | Contents |
|---|---|
| `gator/` | Reusable library: models, memory backends, optimizers, tokenization, data pipeline, training loop, CLI |
| `gator/data/`, `gator/preparation.py` | Dataset loading and preparation (TinyStories, SmolTalk and others) |
| `gator_path/` | Gator-Path packet topology and routing |
| `gator_linear/` | Gator-Linear variants |
| `dual_expert_coordinator/` | Dual-expert coordinator prototype |
| `*.py` (root) | Training scripts, toy models, benchmarks, checkpoint analysis and Transformer baselines |
| `Runbooks/` | Runbooks, the run registry, launch scripts and the tokenizer |
| `Papers/` | Design specifications, experiment write-ups and configuration guides |
| `*.ipynb` | Colab notebooks |
| `tests/` | Unit tests |

## Documentation

- [Architecture specification](ARCHITECTURE.md): equations, tensor shapes, parameter accounting
- [Language trainer arguments](GATOR_LANGUAGE_TRAINER_ARGUMENTS.md)
- [Training guide](Papers/GATOR_TRAINING_GUIDE.md) and [configuration how-to](Papers/GATOR_CONFIG_HOWTO.md)
- [Gator-Path design specification](Papers/GATOR_PATH_DESIGN_SPEC.md)
- [Gator-Linear design](Papers/GATOR_LINEAR_DESIGN.md)
- [Preliminary findings](Papers/GTR_internal_findings.md)
- [Runbooks](Runbooks/README.md)

[README_ORIGINAL.md](README_ORIGINAL.md) is the working README from development, with detailed
status notes, results tables and commands. It and other documents may refer to checkpoints, logs
and branches that are not part of this release.

## Getting started

The code needs Python 3.10+ and PyTorch. There is no packaging or pinned requirements file; install
PyTorch for your platform, then run scripts from the repository root.

```bash
python gtr_toy.py
```

```bash
python -m pytest tests
```

Launch scripts under `Runbooks/colab_cells/` that download gated models or datasets expect a
Hugging Face token in the `HF_TOKEN` environment variable.

## What is not included

Model checkpoints, training logs, result tables and bundled Colab archives are not part of this
repository.

## License

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, modify and
share this code for personal, research and other noncommercial purposes. Commercial use is not
permitted.

Required Notice: Copyright (c) 2026 RiujinZero (https://github.com/RiujinZero/GTR-oss)
