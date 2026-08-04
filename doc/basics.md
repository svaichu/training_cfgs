# Config basics

```python
from training_cfgs import Config
```

`Config` is a fluent, self-learning config builder for training runs (dataset,
training hyperparameters, etc.). It groups fields under top-level categories
(e.g. `dataset`, `training`), learns each field's name/type/default from a
YAML/JSON file or an explicit `define()` call, and generates group methods
dynamically (`config.dataset(...)`, `config.training(...)`).

Unlike a plain dict or `argparse.Namespace`, a group or field must exist —
either loaded from a file or registered with `define()` — before it can be
set through the fluent API. Setting an unknown group or unknown field raises
a clear error instead of silently creating it.

This page covers the core workflow: building a config, loading it, reading
and updating values, CLI overrides, and exporting — no hyperparameter search
space involved. If you're setting up a W&B sweep or an Optuna study, start
here to learn the basics, then move on to
[HPO bounds & W&B sweeps](wandb_sweeps.md) or [Optuna integration](optuna.md).

## Quick start

```python
from training_cfgs import Config

cfg = Config()
cfg.define("dataset", "name", default="oxe")
cfg.define("dataset", "batch_size", default=32)
cfg.define("training", "learning_rate", default=1e-4)

cfg.dataset(name="oxe", batch_size=64).training(learning_rate=3e-4)

cfg.dataset.name          # "oxe"
cfg.dataset.batch_size    # 64
cfg.groups()               # ["dataset", "training"]
cfg.fields("dataset")      # ["name", "batch_size"]
```

## Loading from a file

Loading a YAML/JSON file teaches the config its groups, field names, and each
field's `type`/`default`. A plain value (e.g. `batch_size: 32`) is enough —
`type` is inferred and `default` is set to the value itself:

```yaml
# config.yaml
dataset:
  name: oxe
  batch_size: 32
  shuffle: true

training:
  learning_rate: 1.0e-4
  optimizer: adam
  num_epochs: 100
```

```python
cfg = Config.from_file("config.yaml")   # dispatches on extension (.yaml/.yml/.json)
# or explicitly:
cfg = Config.from_yaml("config.yaml")
cfg = Config.from_json("config.json")
cfg = Config.from_dict({"dataset": {"name": "oxe"}})
```

`pyyaml` and `optuna` are core dependencies (`pip install training-cfgs`
pulls in both), so `from_yaml`/Optuna support work out of the box with no
extras to remember.

`Config.from_yaml()` (or `from_file()`/`from_json()`/`from_dict()`) is the
primary, recommended way to build a config — it's the workflow used
throughout the HPO pages ([W&B sweeps](wandb_sweeps.md),
[Optuna](optuna.md)) and the one to reach for by default. `define()` and
`add_argument()` (below) exist for building a schema entirely in Python when
there's no config file to load, e.g. quick scripts or tests.

Runnable example: [`examples/load_from_yaml_example.py`](../examples/load_from_yaml_example.py).

## Reading and updating values

```python
cfg.dataset.name              # attribute access
cfg.dataset["name"]           # item access
"name" in cfg.dataset         # membership check
cfg.dataset.to_dict()         # {"name": "oxe", "batch_size": 32, ...}

cfg.dataset(name="libero")    # update; returns the parent Config, so calls chain
cfg.dataset(name="libero").training(learning_rate=5e-4)
```

Setting an unregistered group or field raises:

```python
cfg.dataset(missing_field=1)
# KeyError: Unknown field 'dataset.missing_field'; define it with
# Config.define('dataset', 'missing_field', ...) or load it from a file first

cfg.unknown_group(x=1)
# AttributeError: Unknown group 'unknown_group'; define it with
# Config.define('unknown_group', ...) or load it from a file first
```

`cfg.clone()` returns an independent copy that preserves the full schema
(types, distributions) — useful whenever you need to hand out a per-trial
or per-run variant without mutating the original (this is how
`get_current_from_optuna()` and `from_optuna_params()` build their returned
configs — see [Optuna integration](optuna.md)).

## CLI overrides (argparse)

Every known field is exposed as a dotted, typed `--<group>.<field>` command-line
option — the standard training-script pattern, and the same keys the W&B sweep
export uses, so a `wandb agent` command line
(`python train.py --training.learning_rate=0.001`) parses directly.
Precedence is **defaults < config file < CLI**: options the user didn't pass
keep their config values.

Runnable examples: [`examples/load_from_yaml_example.py`](../examples/load_from_yaml_example.py)
(`Config.from_yaml()` then `parse_args()` directly, no `--config` flag),
[`examples/train_cli_example.py`](../examples/train_cli_example.py)
(`Config.from_cli()` loading a YAML file) and
[`examples/train_cli_programmatic_example.py`](../examples/train_cli_programmatic_example.py)
(`add_argument()` building the schema entirely in Python).

**Loading from YAML/JSON already wires up the CLI — no extra step.** Every
`from_yaml`/`from_json`/`from_file`/`from_dict` call ends by resyncing
`cfg.parser` with the fields it just learned, so `cfg.parse_args()` works
immediately:

```python
cfg = Config.from_yaml("config.yaml")
cfg.parse_args()   # --dataset.*, --training.* etc. are already there
```

`set_distribution` resyncs the parser too (so a `CategoricalDistribution`
becomes an argparse `choices` option right away — see
[HPO bounds & W&B sweeps](wandb_sweeps.md)), and `add_argument()`/`define()`
extend the same parser incrementally as new fields are registered. The only
time you call `add_arguments(parser)` yourself is to compose config options
into a *separate*, externally-owned parser (see below).

The one-liner for a train script is `Config.from_cli()`, which handles
`--config <file>` plus overrides:

```python
# train.py
cfg = Config.from_cli(default_config="config.yaml")
```

```bash
python train.py --config other.yaml \
    --training.learning_rate 1e-3 \
    --dataset.batch_size 64 \
    --dataset.shuffle false
```

Values are converted using each field's learned `type`:

- `int` / `float` / `str` — converted directly; passing a non-numeric value to
  an `int` field is an argparse error.
- `bool` — accepts a bare flag (`--dataset.shuffle`) or an explicit value
  (`--dataset.shuffle false`, `--dataset.shuffle=True`; `1/0/yes/no/on/off`
  also work).
- `list` — a JSON array (`'["a", "b"]'`) or comma-separated string
  (`wrist,front` → `["wrist", "front"]`, `1,2,3` → `[1, 2, 3]`).
- `dict` — a JSON object (`'{"warmup": 10}'`).

A field with a `CategoricalDistribution` becomes an argparse `choices`
option, so passing a value outside the set is rejected at parse time.
Overriding a field never touches its schema `default` — only the current
value changes.

### Building a config straight from argparse-style calls

A `Config` owns its own `argparse.ArgumentParser` from the moment it's
constructed (`cfg.parser`), and keeps it in sync with the schema as fields
are added. `add_argument()` mirrors `argparse.ArgumentParser.add_argument`
but takes a dotted `"group.field"` name, so it registers the field *and*
exposes it on the command line in one call — no `Config.from_file(...)` or
separate `add_arguments(parser)` step needed for a config built entirely in
Python:

```python
cfg = Config(description="Train a policy")
cfg.add_argument("dataset.name", default="oxe")
cfg.add_argument("dataset.batch_size", default=32)
cfg.add_argument("training.learning_rate", default=1e-4, type=float)
cfg.add_argument("training.optimizer", default="adam", choices=["adam", "sgd"])

cfg.parse_args()   # parses sys.argv against cfg.parser and applies overrides
```

`type` accepts a Python type (`int`, `float`, `bool`, `list`, `dict`) or the
schema's string name; `choices=[...]` is shorthand for
`set_distribution(..., CategoricalDistribution(choices))` (see
[HPO bounds & W&B sweeps](wandb_sweeps.md)) that also becomes an argparse
`choices` option; `help="..."` overrides the auto-generated
`(type) default: ...` help text. A leading `--` on the name is optional
(`add_argument("--training.learning_rate", ...)` works too).

Because loading a file (`from_file`/`from_yaml`/`from_json`) and
`set_distribution` also resync `cfg.parser`, `cfg.parse_args()` works
immediately after any of them — `add_arguments(parser)` is only needed when
composing config options into a *separate* parser:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="run0")

cfg = Config.from_file("config.yaml")
cfg.add_arguments(parser)              # adds --dataset.*, --training.*, ...
args = parser.parse_args()
cfg.apply_args(args)                   # applies dotted keys; ignores run_name
```

Or, when the config owns the whole command line:

```python
cfg = Config.from_file("config.yaml")
cfg.parse_args()                       # parses sys.argv against cfg.parser, applies
cfg.parse_args(strict=False)           # tolerate argv entries meant for others
```

`apply_args` also accepts a plain dict of dotted keys (string values are
coerced through the field's type), which is convenient for applying
`wandb.config` inside a sweep run:

```python
cfg.apply_args(dict(wandb.config))     # {"training.learning_rate": 0.0007, ...}
```

## Exporting

```python
cfg.to_dict()             # nested dict; sweepable fields export as {type, default, distribution}
cfg.save("out.yaml")      # or "out.json"
cfg.clone()                # independent copy, same schema
```

## API summary

| Member | Description |
|---|---|
| `Config(description=None)` | Empty config with its own `argparse.ArgumentParser`; groups/fields must be `define()`d/`add_argument()`d or loaded before use |
| `define(group, field, default=None, type=None, **extra)` | Register a field, creating its group if needed |
| `add_argument(name, default=None, type=None, help=None, choices=None, **extra)` | argparse-style shorthand: `define()` a dotted `"group.field"` and expose it on `cfg.parser` in one call |
| `parser` | The config's internal `argparse.ArgumentParser`, resynced whenever fields/distributions change |
| `groups()` / `fields(group)` | List known group / field names |
| `schema(group, field)` | Return the field's `FieldSpec` (`type`, `default`, `distribution`, `help`) |
| `Config.from_dict(data)` / `from_yaml(path)` / `from_json(path)` / `from_file(path)` | Load groups/fields from a dict or file |
| `Config.from_cli(argv=None, default_config=None, description=None)` | Load `--config <file>` and apply `--<group>.<field>` overrides |
| `add_arguments(parser, groups=None)` | Add typed `--<group>.<field>` options to a *separate* `argparse` parser |
| `apply_args(args)` | Apply dotted overrides from a parsed namespace or dict |
| `parse_args(argv=None, parser=None, strict=True)` | Parse against `cfg.parser` (or a given `parser`) and apply overrides |
| `to_dict()` | Export the current config as a nested dict |
| `clone()` | Independent copy, preserving schema |
| `save(path)` | Write to `.yaml`/`.yml`/`.json` |

`config.<group>(**fields)` (e.g. `config.dataset(...)`) updates a group and
returns the parent `Config` for chaining; `config.<group>.<field>` reads the
current value.

For attaching a search space (`set_distribution`) and exporting it, see
[HPO bounds & W&B sweeps](wandb_sweeps.md) and [Optuna integration](optuna.md).
