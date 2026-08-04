# training_cfgs

Standalone config manager for training runs: a fluent, self-learning
`Config` object with typed CLI overrides, W&B sweep export, and two-way
[Optuna](https://optuna.org/) integration — all driven by the same schema.

```bash
pip install -e .          # pulls in pyyaml + optuna
pip install -e ".[dev]"   # + pytest, for running the test suite
```

```python
from training_cfgs import Config

cfg = Config.from_file("config.yaml")
cfg.set_bounds("training", "learning_rate", min=1e-5, max=1e-2, log=True)
cfg.set_values("training", "optimizer", ["adam", "sgd"])

sweep = cfg.to_sweep()
```

See [Config basics](basics.md) for loading, CLI overrides, and exporting;
[HPO bounds & W&B sweeps](wandb_sweeps.md) for attaching a search space and
exporting it to W&B; and [Optuna integration](optuna.md) for running the
search itself — or jump straight to the [API reference](api.md).

```{toctree}
:maxdepth: 2
:hidden:

basics
wandb_sweeps
optuna
api
```
