"""Example: run an Optuna hyperparameter search over a Config's sweepable fields.

`set_distribution` attaches a native `optuna.distributions` object to a
field -- the exact same schema used for W&B sweep export (see `main.py` /
`to_sweep`), so a config only needs to be annotated once to support both.
`to_optuna_distributions()` exposes the search space up front;
`get_current_from_optuna(trial)` returns a new, fully-populated `Config` per trial without
mutating the original.

    python examples/optuna_example.py
"""

from pathlib import Path

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

from training_cfgs import Config

DEFAULT_CONFIG = Path(__file__).parent / "sample_config.yaml"


def build_config() -> Config:
    cfg = Config.from_file(DEFAULT_CONFIG)
    cfg.set_distribution("training", "learning_rate", FloatDistribution(1e-5, 1e-2, log=True))
    cfg.set_distribution("training", "optimizer", CategoricalDistribution(["adam", "sgd"]))
    cfg.set_distribution("dataset", "batch_size", IntDistribution(8, 128, step=8))
    return cfg


def fake_train(trial_cfg: Config) -> float:
    """Stand-in for a real training loop: pretend the optimum is lr=1e-3, batch_size=64."""
    lr_term = (trial_cfg.training.learning_rate - 1e-3) ** 2
    batch_term = (trial_cfg.dataset.batch_size - 64) ** 2 / 1e5
    optimizer_penalty = 0.0 if trial_cfg.training.optimizer == "adam" else 0.05
    return lr_term + batch_term + optimizer_penalty


def main() -> None:
    cfg = build_config()

    print("Search space:", cfg.to_optuna_distributions())

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = cfg.get_current_from_optuna(trial)
        return fake_train(trial_cfg)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    best_cfg = cfg.from_optuna_study(study)
    print(best_cfg)
    best_cfg.save("best_config.yaml")


if __name__ == "__main__":
    main()
