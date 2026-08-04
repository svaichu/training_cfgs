# training_cfgs

Standalone config manager migrated from `svaichu/robotdataset`: a fluent,
self-learning `Config` object for training runs, with typed CLI overrides,
W&B sweep export, and two-way [Optuna](https://optuna.org/) integration —
all driven by the same schema.

📖 **[Full documentation on Read the Docs](https://training-cfgs.readthedocs.io/)**

## Install

`pyyaml` and `optuna` are core dependencies — a plain install always gets
YAML config loading and Optuna support, no extras to remember:

```bash
pip install -e .          # pulls in pyyaml + optuna
pip install -e ".[dev]"   # + pytest, for running the test suite
```

## Quick start

```python
from training_cfgs import Config, distributions, study

cfg = Config.from_file("config.yaml")
cfg.set_distribution("training", "learning_rate", distributions.FloatDistribution(1e-5, 1e-2, log=True))
cfg.set_distribution("training", "optimizer", distributions.CategoricalDistribution(["adam", "sgd"]))

# W&B sweep export
sweep = cfg.to_sweep()

# CLI overrides: python train.py --training.learning_rate 1e-3
cfg = Config.from_cli(default_config="config.yaml")

# Optuna search, same schema as the sweep export above
study = study.create_study(direction="minimize")
cfg.single_objective_optimization(study, train, n_trials=50)
best_cfg = cfg.best_from_optuna(study)
```

## Docs

The full guide lives on **[Read the
Docs](https://training-cfgs.readthedocs.io/)**, built from the same sources
in this repo: [`doc/basics.md`](doc/basics.md) (loading, CLI overrides,
exporting), [`doc/wandb_sweeps.md`](doc/wandb_sweeps.md) (HPO bounds, W&B
sweep export), [`doc/optuna.md`](doc/optuna.md) (Optuna integration) — and
[`examples/`](examples/) for runnable scripts.
