# training_cfgs

Standalone config manager migrated from `svaichu/robotdataset`: a fluent,
self-learning `Config` object for training runs, with typed CLI overrides,
W&B sweep export, and two-way [Optuna](https://optuna.org/) integration —
all driven by the same schema.

## Install

`pyyaml` and `optuna` are core dependencies — a plain install always gets
YAML config loading and Optuna support, no extras to remember:

```bash
pip install -e .          # pulls in pyyaml + optuna
pip install -e ".[dev]"   # + pytest, for running the test suite
```

## Quick start

```python
from training_cfgs import Config

cfg = Config.from_file("config.yaml")
cfg.set_bounds("training", "learning_rate", min=1e-5, max=1e-2, log=True)
cfg.set_values("training", "optimizer", ["adam", "sgd"])

# W&B sweep export
sweep = cfg.to_sweep()

# CLI overrides: python train.py --training.learning_rate 1e-3
cfg = Config.from_cli(default_config="config.yaml")

# Optuna search, same schema as the sweep export above
def objective(trial):
    trial_cfg = cfg.suggest(trial)
    return train(trial_cfg)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_cfg = cfg.from_optuna_study(study)
```

See [`doc/config.md`](doc/config.md) (or the hosted [Read the Docs
site](https://training-cfgs.readthedocs.io/), once connected) for the full
guide (loading, CLI overrides, W&B sweeps, Optuna) and
[`examples/`](examples/) for runnable scripts.
