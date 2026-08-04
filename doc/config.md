# Config

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

A field may also be spelled out explicitly as a spec dict —
`{type: ..., default: ..., distribution: ..., help: ...}` — which is how a
`distribution` (see below) gets declared directly in the file instead of
being attached afterward. `type` itself is optional in a spec dict: it's
inferred from `default` exactly like a plain value would be, so
`{default: 1.0e-4, distribution: {name: FloatDistribution, attributes: {low: 1.0e-5, high: 1.0e-2}}}`
and
`{type: float, default: 1.0e-4, distribution: {name: FloatDistribution, attributes: {low: 1.0e-5, high: 1.0e-2}}}`
are equivalent. Any one of `type`/`distribution`/`help` present is enough
for the loader to treat the mapping as a spec dict rather than a literal
`dict`-typed value.

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
or per-run variant without mutating the original (this is how `get_current_from_optuna()`
and `from_optuna_params()` build their returned configs, see below).

## Hyperparameter opt settings (`optuna.distributions`)

A field's file entry doesn't need to carry any search-space info. A field
becomes sweepable by attaching a native
[`optuna.distributions`](https://optuna.readthedocs.io/en/stable/reference/distributions.html)
object — `FloatDistribution`, `IntDistribution`, `CategoricalDistribution`,
etc. — directly, so there's no separate min/max/values schema to keep in
sync with Optuna's own types. There are two equally valid ways to attach
one, and both produce the exact same schema, which is what makes a field
eligible for W&B sweep export *and* Optuna search spaces:

**1. Load a plain config, then attach a distribution in Python:**

```python
from optuna.distributions import CategoricalDistribution, FloatDistribution

cfg = Config.from_file("config.yaml")   # plain values only
cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2))
cfg.set_distribution("training", "optimizer", CategoricalDistribution(["adam", "sgd"]))

# equivalent group-level shorthand
cfg.training.set_distribution("learning_rate", FloatDistribution(1e-5, 1e-2))
cfg.training.set_distribution("optimizer", CategoricalDistribution(["adam", "sgd"]))
```

**2. Declare `distribution` directly in the file** — no Python step
needed; the field is sweepable the moment the file is loaded. The dict
mirrors Optuna's own `distribution_to_json`/`json_to_distribution` shape:
`{name: <distribution class name>, attributes: {...constructor kwargs...}}`:

```yaml
# config.yaml
training:
  learning_rate:
    default: 1.0e-4        # type inferred as float; "type: float" also works
    distribution:
      name: FloatDistribution
      attributes: {low: 1.0e-5, high: 1.0e-2}
  optimizer:
    default: adam
    distribution:
      name: CategoricalDistribution
      attributes: {choices: [adam, sgd]}
  num_epochs: 100           # plain value: fixed, not sweepable
```

```python
cfg = Config.from_file("config.yaml")
cfg.schema("training", "learning_rate").distribution   # FloatDistribution(low=1e-05, high=0.01, ...) -- already set
```

See [`examples/sweepable_config.yaml`](../examples/sweepable_config.yaml) /
[`examples/sweepable_config_example.py`](../examples/sweepable_config_example.py)
for a complete runnable version of approach 2, including `log`/`step`.

`FloatDistribution`/`IntDistribution` accept `log=True` (log-uniform
sampling) and `step=...` (discretized range) — the exact same kwargs Optuna
itself uses, since these *are* Optuna's constructors:

```python
from optuna.distributions import FloatDistribution, IntDistribution

cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
cfg.set_distribution("dataset", "batch_size", IntDistribution(8, 256, step=8))
```

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
becomes an argparse `choices` option right away), and `add_argument()`/`define()`
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
`set_distribution(..., CategoricalDistribution(choices))` that also becomes
an argparse `choices` option; `help="..."` overrides the auto-generated
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

The sweep-conversion entrypoint accepts the same overrides before exporting:

```bash
python -m training_cfgs.main config.yaml sweep.yaml \
    --method bayes --training.num_epochs 200
```

## Exporting

```python
cfg.to_dict()             # nested dict; sweepable fields export as {type, default, distribution}
cfg.save("out.yaml")      # or "out.json"
cfg.clone()                # independent copy, same schema
```

## W&B sweep export

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

## Optuna integration

A field's search-space schema *is* Optuna's own
[`optuna.distributions`](https://optuna.readthedocs.io/en/stable/reference/distributions.html)
type — `FieldSpec.distribution` is handed to Optuna as-is, with no
bounds/values translation layer to maintain by hand — so it drives both W&B
sweep export and two-way [Optuna](https://optuna.org/) compatibility from
one native representation. `optuna` is a core dependency, installed
automatically with the package. All of the logic below lives in
[`training_cfgs/optuna_compat.py`](../training_cfgs/optuna_compat.py) and is
exposed as `Config` methods; runnable end-to-end example:
[`examples/optuna_example.py`](../examples/optuna_example.py).

### The interface, precisely

There are two directions of translation. `Config` never imports
`optuna.Trial`/`optuna.Study` as concrete types — it only calls
`trial._suggest(name, distribution)` (Optuna's own generic entry point,
which `suggest_float`/`suggest_int`/`suggest_categorical` call internally)
and reads `study.best_params`, so any object with that shape works,
including Optuna's own `FrozenTrial`/mocks in tests.

**1. `Config` schema → `optuna.distributions` (the search space).**
`to_optuna_distributions(groups=None)` walks every field whose `FieldSpec`
`is_sweepable()` (has a `distribution` attached — see the previous section)
and collects `spec.distribution` directly, keyed `"group.field"`:

```python
distributions = cfg.to_optuna_distributions()
# {"training.learning_rate": FloatDistribution(low=1e-5, high=1e-2, log=True, step=None),
#  "training.optimizer": CategoricalDistribution(choices=("adam", "sgd"))}
```

Non-sweepable fields (plain values, no `distribution`) are skipped entirely
— they're not part of the search space. This dict on its own is useful for
samplers/APIs that want the search space up front
(`study.enqueue_trial(params, skip_if_exists=True)`, `study.add_trial(...)`,
distribution-aware samplers) without running a trial.

**2. `optuna.Trial` → a fully-populated `Config` (per-trial values).**
`get_current_from_optuna(trial, groups=None)` is what an objective function calls each
trial. For every sweepable field it calls `trial._suggest(key,
spec.distribution)` and writes the result into a **clone** of the config —
the original `cfg` and every non-sweepable field (`num_epochs` in the
example below) are left untouched:

```python
from optuna.distributions import CategoricalDistribution, FloatDistribution

cfg = Config.from_file("config.yaml")
cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
cfg.set_distribution("training", "optimizer", CategoricalDistribution(["adam", "sgd"]))

def objective(trial: optuna.Trial) -> float:
    trial_cfg = cfg.get_current_from_optuna(trial)          # -> Config, e.g. lr=0.0032, optimizer="sgd"
    return train(trial_cfg)                 # trial_cfg.training.num_epochs == cfg's original value

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)      # calls objective(trial) once per trial
```

Because `get_current_from_optuna()` clones (`Config.clone()`) instead of mutating `cfg` in
place, the same `cfg` object is safe to reuse across every trial in the
study — nothing needs to be reset between calls.

**Or skip the objective closure entirely: `single_objective_optimization`.**
`single_objective_optimization(study, train, groups=None, **optimize_kwargs)`
wraps the `objective`/`study.optimize` boilerplate above into one call —
`train` is any callable that takes a per-trial `Config` and returns the
float `study.optimize` expects; everything else (`n_trials`, `timeout`,
`callbacks`, ...) is forwarded straight through:

```python
def train(trial_cfg: Config) -> float:
    ...

study = optuna.create_study(direction="minimize")
cfg.single_objective_optimization(study, train, n_trials=50)
```

This is exactly the two lines above (`def objective(trial): ...` +
`study.optimize(objective, n_trials=50)`) with the closure written for you —
reach for the explicit form instead when the objective needs to do more than
call `train(trial_cfg)` (e.g. `trial.report(...)`/pruning mid-training).

**3. The winning trial's params → a `Config` (loading the result back).**
`best_from_optuna(study)` reads `study.best_params` (a plain
`{"group.field": value, ...}` dict, Optuna's own trial-recording format) and
applies it onto a clone of `cfg`, exactly like `apply_args` does for
CLI/file overrides. `from_optuna_params(params)` is the same operation
against *any* dotted-key dict — `study.best_params`, `trial.params` from a
specific completed trial, or a dict you built by hand:

```python
best_cfg = cfg.best_from_optuna(study)           # study.best_params
best_cfg = cfg.from_optuna_params(trial.params)  # one specific trial's params
```

Both return a clone with the given `"group.field"` params applied on top of
the original config's other values, so fixed (non-sweepable) fields like
`num_epochs` are carried over unchanged — the round trip is
`cfg.set_distribution` → `study.optimize` → `cfg.best_from_optuna`, ending
with a `Config` of the same shape you started with.

### Validation and edge cases

- A field with no `distribution` attached is simply skipped by
  `to_optuna_distributions`/`get_current_from_optuna` — it's not part of the
  search space, and its value carries over unchanged from the original config.
- `from_optuna_params` raises `ValueError` for a key without a `.`
  (`{"learning_rate": ...}` instead of `{"training.learning_rate": ...}`),
  matching `apply_args`'s strictness for CLI/`wandb.config` overrides.
- `groups=[...]` on any of these methods restricts them to specific
  top-level groups, same as `to_sweep(groups=...)`.

## API summary

| Member | Description |
|---|---|
| `Config(description=None)` | Empty config with its own `argparse.ArgumentParser`; groups/fields must be `define()`d/`add_argument()`d or loaded before use |
| `define(group, field, default=None, type=None, **extra)` | Register a field, creating its group if needed |
| `add_argument(name, default=None, type=None, help=None, choices=None, **extra)` | argparse-style shorthand: `define()` a dotted `"group.field"` and expose it on `cfg.parser` in one call |
| `parser` | The config's internal `argparse.ArgumentParser`, resynced whenever fields/distributions change |
| `groups()` / `fields(group)` | List known group / field names |
| `schema(group, field)` | Return the field's `FieldSpec` (`type`, `default`, `distribution`, `help`) |
| `set_distribution(group, field, distribution)` | Attach an `optuna.distributions.BaseDistribution` to an existing field, making it sweepable |
| `Config.from_dict(data)` / `from_yaml(path)` / `from_json(path)` / `from_file(path)` | Load groups/fields from a dict or file |
| `Config.from_cli(argv=None, default_config=None, description=None)` | Load `--config <file>` and apply `--<group>.<field>` overrides |
| `add_arguments(parser, groups=None)` | Add typed `--<group>.<field>` options to a *separate* `argparse` parser |
| `apply_args(args)` | Apply dotted overrides from a parsed namespace or dict |
| `parse_args(argv=None, parser=None, strict=True)` | Parse against `cfg.parser` (or a given `parser`) and apply overrides |
| `to_dict()` | Export the current config as a nested dict |
| `clone()` | Independent copy, preserving schema |
| `save(path)` | Write to `.yaml`/`.yml`/`.json` |
| `to_sweep(method="bayes", metric=None, groups=None)` / `to_sweep_file(path, ...)` | Build/write a W&B-compatible sweep config |
| `to_optuna_distributions(groups=None)` | Build a dict of `optuna.distributions`, keyed `"group.field"` |
| `get_current_from_optuna(trial, groups=None)` | New `Config` with sweepable fields set from an `optuna.Trial`'s suggestions |
| `single_objective_optimization(study, train, groups=None, **optimize_kwargs)` | Run `study.optimize` against `train`, building each trial's `Config` automatically |
| `from_optuna_params(params)` | New `Config` with dotted `"group.field"` params (e.g. `trial.params`) applied |
| `best_from_optuna(study)` | New `Config` with a completed `optuna.Study`'s `best_params` applied |

`config.<group>(**fields)` (e.g. `config.dataset(...)`) updates a group and
returns the parent `Config` for chaining; `config.<group>.<field>` reads the
current value.
