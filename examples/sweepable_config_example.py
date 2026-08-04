"""Example: an optuna.distributions object declared directly in the YAML file.

Fields whose spec dict carries a `distribution` are recognized as sweepable
the moment the file is loaded -- no set_distribution() calls needed. `type`
is optional in the spec dict; it's inferred from `default` when omitted
(see sweepable_config.yaml: `dataset.batch_size` and `training.optimizer`
never spell out `type`).

    python examples/sweepable_config_example.py
"""

from pathlib import Path

from training_cfgs import Config

CONFIG_PATH = Path(__file__).parent / "sweepable_config.yaml"


def main() -> None:
    cfg = Config.from_yaml(CONFIG_PATH)
    print(cfg)

    print("\nW&B sweep parameters:")
    print(cfg.to_sweep()["parameters"])

    print("\nOptuna search space:")
    print(cfg.to_optuna_distributions())


if __name__ == "__main__":
    main()
