# Optuna integration

```python
from training_cfgs import Config
```

`Config` has two-way [Optuna](https://optuna.org/) integration: a field's
search-space schema *is* Optuna's own
[`optuna.distributions`](https://optuna.readthedocs.io/en/stable/reference/distributions.html)
type — `FieldSpec.distribution` is handed to Optuna as-is, with no
bounds/values translation layer to maintain by hand — so it drives both
Optuna search spaces and [W&B sweep export](wandb_sweeps.md) from one native
representation. `optuna` is a core dependency, installed automatically with
the package.

If you haven't attached a distribution to a field yet, see
[HPO bounds & W&B sweeps](wandb_sweeps.md) first — everything below assumes
`cfg` already has sweepable fields, built the primary way via
`Config.from_yaml()` with `distribution` declared in the file.

All of the logic below lives in
[`training_cfgs/optuna_compat.py`](../training_cfgs/optuna_compat.py) and is
exposed as `Config` methods; runnable end-to-end example:
[`examples/optuna_example.py`](../examples/optuna_example.py).

## The interface, precisely

There are two directions of translation. `Config` never imports
`optuna.Trial`/`optuna.Study` as concrete types — it only calls
`trial._suggest(name, distribution)` (Optuna's own generic entry point,
which `suggest_float`/`suggest_int`/`suggest_categorical` call internally)
and reads `study.best_params`, so any object with that shape works,
including Optuna's own `FrozenTrial`/mocks in tests.

### 1. `Config` schema → `optuna.distributions` (the search space)

`to_optuna_distributions(groups=None)` walks every field whose `FieldSpec`
`is_sweepable()` (has a `distribution` attached) and collects
`spec.distribution` directly, keyed `"group.field"`:

```python
cfg = Config.from_yaml("config.yaml")   # distribution declared per-field in the file
distributions = cfg.to_optuna_distributions()
# {"training.learning_rate": FloatDistribution(low=1e-5, high=1e-2, log=True, step=None),
#  "training.optimizer": CategoricalDistribution(choices=("adam", "sgd"))}
```

Non-sweepable fields (plain values, no `distribution`) are skipped entirely
— they're not part of the search space. This dict on its own is useful for
samplers/APIs that want the search space up front
(`study.enqueue_trial(params, skip_if_exists=True)`, `study.add_trial(...)`,
distribution-aware samplers) without running a trial.

### 2. `optuna.Trial` → a fully-populated `Config` (per-trial values)

`get_current_from_optuna(trial, groups=None)` is what an objective function
calls each trial. For every sweepable field it calls `trial._suggest(key,
spec.distribution)` and writes the result into a **clone** of the config —
the original `cfg` and every non-sweepable field (`num_epochs` in the
example below) are left untouched:

```python
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution

from training_cfgs import Config

cfg = Config.from_yaml("config.yaml")
cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
cfg.set_distribution("training", "optimizer", CategoricalDistribution(["adam", "sgd"]))

def objective(trial: optuna.Trial) -> float:
    trial_cfg = cfg.get_current_from_optuna(trial)          # -> Config, e.g. lr=0.0032, optimizer="sgd"
    return train(trial_cfg)                 # trial_cfg.training.num_epochs == cfg's original value

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)      # calls objective(trial) once per trial
```

Because `get_current_from_optuna()` clones (`Config.clone()`) instead of
mutating `cfg` in place, the same `cfg` object is safe to reuse across every
trial in the study — nothing needs to be reset between calls.

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

### 3. The winning trial's params → a `Config` (loading the result back)

`best_from_optuna(study)` reads `study.best_params` (a plain
`{"group.field": value, ...}` dict, Optuna's own trial-recording format) and
applies it onto a clone of `cfg`, exactly like `apply_args` does for
CLI/file overrides (see [Config basics](basics.md#cli-overrides-argparse)).
`from_optuna_params(params)` is the same operation against *any* dotted-key
dict — `study.best_params`, `trial.params` from a specific completed trial,
or a dict you built by hand:

```python
best_cfg = cfg.best_from_optuna(study)           # study.best_params
best_cfg = cfg.from_optuna_params(trial.params)  # one specific trial's params
```

Both return a clone with the given `"group.field"` params applied on top of
the original config's other values, so fixed (non-sweepable) fields like
`num_epochs` are carried over unchanged — the round trip is
`cfg.set_distribution` → `study.optimize` → `cfg.best_from_optuna`, ending
with a `Config` of the same shape you started with.

## Full example

Putting it together, starting from a YAML file with distributions declared
inline (the primary `from_yaml` workflow — see
[`examples/sweepable_config.yaml`](../examples/sweepable_config.yaml)):

```python
import optuna
from training_cfgs import Config

cfg = Config.from_yaml("sweepable_config.yaml")

print("Search space:", cfg.to_optuna_distributions())

study = optuna.create_study(direction="minimize")
cfg.single_objective_optimization(study, train, n_trials=30)

print("Best value:", study.best_value)
print("Best params:", study.best_params)

best_cfg = cfg.best_from_optuna(study)
best_cfg.save("best_config.yaml")
```

See [`examples/optuna_example.py`](../examples/optuna_example.py) for a
complete runnable version (with a `set_distribution`-based config, a stand-in
`train` function, and printed output at each step).

## Persisting and resuming a study

`optuna.create_study` keeps everything in memory by default, so the study
(and every trial in it) is lost when the process exits. Pass a `storage` URL
to persist it — SQLite is the simplest option, but any
[SQLAlchemy-compatible URL](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)
(Postgres, MySQL, ...) works, including for multiple processes/machines
optimizing the same study concurrently:

```python
study = optuna.create_study(
    study_name="my_study",
    storage="sqlite:///example.db",
    direction="minimize",
    load_if_exists=True,
)
cfg.single_objective_optimization(study, train, n_trials=50)
```

`study_name` + `load_if_exists=True` make this safe to call repeatedly:
first call creates the study, every later call (a resumed run, a restart
after a crash, another worker) attaches to the existing one instead of
erroring or starting over. To resume later without re-running
`create_study`, `optuna.load_study(study_name=..., storage=...)` loads it
directly; either way, `study.optimize(...)` continues adding trials on top
of the stored history — `study.trials` includes every prior run.

## Visualizing a study

Given a `storage` URL, a study's trial history can be inspected without any
`training_cfgs`-specific code:

- **[`optuna-dashboard`](https://github.com/optuna/optuna-dashboard)**
  (`pip install optuna-dashboard`) — an interactive web UI over the same
  storage URL, including a study that's still running:

  ```bash
  optuna-dashboard sqlite:///example.db
  ```

- **`optuna.visualization`** — in-code plots (Plotly by default, or
  `optuna.visualization.matplotlib` for static figures) for a notebook or
  report:

  ```python
  from optuna.visualization import plot_optimization_history, plot_param_importances

  study = optuna.load_study(study_name="my_study", storage="sqlite:///example.db")
  plot_optimization_history(study).show()
  plot_param_importances(study).show()
  ```

## Validation and edge cases

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
| `to_optuna_distributions(groups=None)` | Build a dict of `optuna.distributions`, keyed `"group.field"` |
| `get_current_from_optuna(trial, groups=None)` | New `Config` with sweepable fields set from an `optuna.Trial`'s suggestions |
| `single_objective_optimization(study, train, groups=None, **optimize_kwargs)` | Run `study.optimize` against `train`, building each trial's `Config` automatically |
| `from_optuna_params(params)` | New `Config` with dotted `"group.field"` params (e.g. `trial.params`) applied |
| `best_from_optuna(study)` | New `Config` with a completed `optuna.Study`'s `best_params` applied |

For attaching distributions in the first place, see
[HPO bounds & W&B sweeps](wandb_sweeps.md).
