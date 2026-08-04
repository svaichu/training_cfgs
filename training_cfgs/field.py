"""Field specifications learned from config files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    IntDistribution,
    distribution_to_json,
    json_to_distribution,
)


def infer_type(value: Any) -> str:
    """Map a Python value to a config field type name."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _default_from_distribution(distribution: BaseDistribution) -> Any:
    if isinstance(distribution, CategoricalDistribution):
        return distribution.choices[0]
    return distribution.low


def _type_from_distribution(distribution: BaseDistribution) -> str:
    if isinstance(distribution, CategoricalDistribution):
        return infer_type(distribution.choices[0])
    return "int" if isinstance(distribution, IntDistribution) else "float"


@dataclass
class FieldSpec:
    """Learned metadata for a single config field.

    `type` and `default` are the required subfields: every field ends up
    with both, inferred from a plain YAML/JSON value if the file didn't
    spell them out explicitly.

    `distribution` is an `optuna.distributions.BaseDistribution` (e.g.
    `FloatDistribution`, `IntDistribution`, `CategoricalDistribution`) that
    makes a field eligible for W&B sweep / Optuna export. The source file
    may declare it directly, but it's typically attached afterward via
    `Config.set_distribution`.
    """

    name: str
    type: str
    default: Any
    distribution: Optional[BaseDistribution] = None
    help: Optional[str] = None

    def is_sweepable(self) -> bool:
        return self.distribution is not None

    @classmethod
    def from_spec_dict(cls, name: str, spec: dict) -> "FieldSpec":
        distribution = spec.get("distribution")
        if isinstance(distribution, dict):
            distribution = json_to_distribution(json.dumps(distribution))

        default = spec.get("default")
        if default is None and distribution is not None:
            default = _default_from_distribution(distribution)

        ftype = spec.get("type")
        if ftype is None:
            if default is not None:
                ftype = infer_type(default)
            elif distribution is not None:
                ftype = _type_from_distribution(distribution)
            else:
                ftype = "str"

        return cls(
            name=name,
            type=ftype,
            default=default,
            distribution=distribution,
            help=spec.get("help"),
        )

    def to_dict(self) -> dict:
        out: dict = {"type": self.type, "default": self.default}
        if self.distribution is not None:
            out["distribution"] = json.loads(distribution_to_json(self.distribution))
        return out
