# HPO bounds & W&B sweeps

```python
from training_cfgs import Config
```

This page covers giving fields a hyperparameter search space (bounds/values)
and exporting that space as a [W&B sweep](https://docs.wandb.ai/guides/sweeps/)
config. If you haven't seen `Config`'s basic loading/CLI/export workflow yet,
start with [Config basics](basics.md) — this page assumes it.

`Config.from_yaml()` is the primary workflow for HPO configs: define the
search space once in the YAML file (or attach it in Python after loading),
and the same schema drives both W&B sweep export and
[Optuna](optuna.md) — no separate min/max/values schema to keep in sync.

## Attaching a search space (`optuna.distributions`)

A field's file entry doesn't need to carry any search-space info. A field
becomes sweepable by attaching a native
[`optuna.distributions`](https://optuna.readthedocs.io/en/stable/reference/distributions.html)
object — `FloatDistribution`, `IntDistribution`, `CategoricalDistribution`,
etc. — directly. `training_cfgs` reuses Optuna's own distribution types as
the bounds format rather than inventing a new one, which is what makes a
field eligible for W&B sweep export *and* Optuna search spaces at once.
There are two equally valid ways to attach one:

**1. Declare `distribution` directly in the YAML file (recommended).**
This is the primary, `from_yaml`-first workflow: no Python step needed, and
the field is sweepable the moment the file is loaded. The dict mirrors
Optuna's own `distribution_to_json`/`json_to_distribution` shape:
`{name: <distribution class name>, attributes: {...constructor kwargs...}}`:

```yaml
# config.yaml
dataset:
  name: oxe
  batch_size:
    default: 32               # type inferred as int
    distribution:
      name: IntDistribution
      attributes: {low: 8, high: 128, step: 8}

training:
  learning_rate:
    default: 1.0e-4           # type inferred as float; "type: float" also works
    distribution:
      name: FloatDistribution
      attributes: {low: 1.0e-5, high: 1.0e-2, log: true}
  optimizer:
    default: adam
    distribution:
      name: CategoricalDistribution
      attributes: {choices: [adam, sgd]}
  num_epochs: 100              # plain value: fixed, not sweepable
```

```python
cfg = Config.from_yaml("config.yaml")
cfg.schema("training", "learning_rate").distribution   # FloatDistribution(low=1e-05, high=0.01, log=True, ...)
```

A field may also be spelled out as a spec dict without a `distribution` key
— `{type: ..., default: ..., help: ...}` — for a plain (non-sweepable)
field written in long form. `type` itself is always optional in a spec
dict: it's inferred from `default` exactly like a plain value would be. Any
one of `type`/`distribution`/`help` present is enough for the loader to
treat the mapping as a spec dict rather than a literal `dict`-typed value.

See [`examples/sweepable_config.yaml`](../examples/sweepable_config.yaml) /
[`examples/sweepable_config_example.py`](../examples/sweepable_config_example.py)
for a complete runnable version of this approach, including `log`/`step`.

**2. Load a plain config, then attach a distribution in Python.** Useful
when the search space is decided at run time rather than baked into the
file:

```python
from optuna.distributions import CategoricalDistribution, FloatDistribution

cfg = Config.from_yaml("config.yaml")   # plain values only
cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
cfg.set_distribution("training", "optimizer", CategoricalDistribution(["adam", "sgd"]))

# equivalent group-level shorthand
cfg.training.set_distribution("learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
cfg.training.set_distribution("optimizer", CategoricalDistribution(["adam", "sgd"]))
```

`FloatDistribution`/`IntDistribution` accept `log=True` (log-uniform
sampling) and `step=...` (discretized range) — the exact same kwargs Optuna
itself uses, since these *are* Optuna's constructors:

```python
from optuna.distributions import FloatDistribution, IntDistribution

cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
cfg.set_distribution("dataset", "batch_size", IntDistribution(8, 256, step=8))
```

`set_distribution` resyncs `cfg.parser` too, so a `CategoricalDistribution`
immediately becomes an argparse `choices` option (see
[CLI overrides](basics.md#cli-overrides-argparse)).

## Exporting a W&B sweep

Fields with a `FloatDistribution`/`IntDistribution` become continuous
ranges, fields with a `CategoricalDistribution` become discrete/categorical
choices, and any other field is exported as a fixed `value` from the
current config:

```python
sweep = cfg.to_sweep(
    method="bayes",
    metric={"name": "loss", "goal": "minimize"},
    groups=["training"],   # optional: restrict to specific groups
)
cfg.to_sweep_file("sweep.yaml", method="bayes", metric={"name": "loss", "goal": "minimize"})
```

```python
sweep["parameters"]["training.learning_rate"]  # {"min": 1e-5, "max": 1e-2}
sweep["parameters"]["training.optimizer"]      # {"values": ["adam", "sgd"]}
sweep["parameters"]["training.num_epochs"]     # {"value": 100}  (fixed, not swept)
```

The same entrypoint is available from the command line, converting a config
file straight into a sweep file:

```bash
python -m training_cfgs.main config.yaml sweep.yaml \
    --method bayes --training.num_epochs 200
```

## End-to-end: config → sweep → training script

Because the search-space schema also drives CLI parsing (every field is a
`--<group>.<field>` option — see [Config basics](basics.md#cli-overrides-argparse)),
a `wandb agent` run and a plain training script share the same override
mechanism:

```python
# make_sweep.py
from training_cfgs import Config

cfg = Config.from_yaml("config.yaml")   # distributions declared in the file
cfg.to_sweep_file("sweep.yaml", method="bayes", metric={"name": "loss", "goal": "minimize"})
```

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>   # launches runs like: python train.py --training.learning_rate=0.0032 ...
```

```python
# train.py
cfg = Config.from_cli(default_config="config.yaml")   # picks up the agent's --training.learning_rate=...
train(cfg)
```

If you're running the sweep loop yourself instead of via `wandb agent`
(e.g. inside a notebook or a custom launcher), `apply_args` applies
`wandb.config` directly:

```python
cfg.apply_args(dict(wandb.config))     # {"training.learning_rate": 0.0007, ...}
```

## API summary

| Member | Description |
|---|---|
| `schema(group, field)` | Return the field's `FieldSpec` (`type`, `default`, `distribution`, `help`) |
| `set_distribution(group, field, distribution)` | Attach an `optuna.distributions.BaseDistribution` to an existing field, making it sweepable |
| `to_sweep(method="bayes", metric=None, groups=None)` | Build a W&B-compatible sweep config dict |
| `to_sweep_file(path, method="bayes", metric=None, groups=None)` | Write a W&B-compatible sweep config to `.yaml`/`.json` |

For running the search itself rather than exporting it to W&B, see
[Optuna integration](optuna.md).
