"""Two-way Optuna compatibility for `Config`.

A field's search space *is* an `optuna.distributions.BaseDistribution` —
attached via `Config.set_distribution(group, field, distribution)` or
declared directly in a config file — so there's no separate bounds/values
schema to translate; `FieldSpec.distribution` is handed to Optuna as-is.

Building a search space and running a study::

    distributions = cfg.to_optuna_distributions()   # optuna.distributions, keyed 'group.field'

    def objective(trial):
        trial_cfg = cfg.suggest(trial)               # sweepable fields replaced with trial suggestions
        return train(trial_cfg)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

Loading the winning config back onto the schema::

    best_cfg = cfg.from_optuna_study(study)           # study.best_params
    best_cfg = cfg.from_optuna_params(trial.params)    # any dotted-key params dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from optuna.distributions import CategoricalDistribution

from .field import FieldSpec

if TYPE_CHECKING:
    from .config import Config


def to_sweep_params(spec: FieldSpec, current_value: Any) -> dict:
    """Convert one field's schema into a W&B sweep `parameters` entry.

    Non-sweepable fields (no `distribution`) export as a fixed `value` from
    the current config.
    """
    distribution = spec.distribution
    if distribution is None:
        return {"value": current_value}
    if isinstance(distribution, CategoricalDistribution):
        return {"values": list(distribution.choices)}
    params: dict = {"min": distribution.low, "max": distribution.high}
    if distribution.log:
        params["log"] = True
    if distribution.step is not None:
        params["step"] = distribution.step
    return params


def to_optuna_distributions(config: "Config", groups: Optional[list[str]] = None) -> dict:
    """Build a dict of `optuna.distributions`, keyed `'group.field'`, from sweepable fields.

    Suitable for `optuna.study.Study.enqueue_trial` / `add_trial` / distribution-aware
    samplers that need the search space up front.
    """
    distributions: dict = {}
    for group, fields in config._schema.items():
        if groups is not None and group not in groups:
            continue
        for field, spec in fields.items():
            if spec.is_sweepable():
                distributions[f"{group}.{field}"] = spec.distribution
    return distributions


def suggest(config: "Config", trial: Any, groups: Optional[list[str]] = None) -> "Config":
    """Return a new `Config` with every sweepable field replaced by `trial`'s suggestion.

    Non-sweepable fields keep their current values unchanged. Use inside an
    Optuna objective function::

        def objective(trial):
            trial_cfg = cfg.suggest(trial)
            return train(trial_cfg)
    """
    cfg = config.clone()
    for group, fields in config._schema.items():
        if groups is not None and group not in groups:
            continue
        for field, spec in fields.items():
            if spec.is_sweepable():
                key = f"{group}.{field}"
                # `Trial._suggest` is Optuna's own generic entry point for an arbitrary
                # `BaseDistribution` — what `suggest_float`/`suggest_int`/`suggest_categorical`
                # call internally, and the only way to suggest from a distribution object
                # without re-deriving low/high/choices by hand.
                cfg._set_field_strict(group, field, trial._suggest(key, spec.distribution))
    return cfg


def from_optuna_params(config: "Config", params: dict) -> "Config":
    """Return a new `Config` with dotted `'group.field'` params (e.g. `trial.params`) applied."""
    cfg = config.clone()
    for key, value in params.items():
        if "." not in key:
            raise ValueError(f"Optuna param key must be 'group.field', got {key!r}")
        group, _, field = key.partition(".")
        cfg._set_field_strict(group, field, value)
    return cfg


def from_optuna_study(config: "Config", study: Any) -> "Config":
    """Return a new `Config` with the winning params from a completed `optuna.Study` applied."""
    return from_optuna_params(config, study.best_params)
