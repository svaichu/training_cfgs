import json

import pytest

from training_cfgs import Config


def test_fluent_builder_chains_and_learns_types():
    cfg = Config()
    cfg.define("dataset", "name", default="oxe")
    cfg.define("dataset", "batch_size", default=0)
    cfg.define("training", "learning_rate", default=0.0)
    cfg.define("training", "shuffle", default=False)
    cfg.dataset(name="oxe", batch_size=32).training(learning_rate=1e-4, shuffle=True)

    assert cfg.groups() == ["dataset", "training"]
    assert cfg.dataset.name == "oxe"
    assert cfg.dataset.batch_size == 32
    assert cfg.training.learning_rate == 1e-4

    assert cfg.schema("dataset", "batch_size").type == "int"
    assert cfg.schema("dataset", "name").type == "str"
    assert cfg.schema("training", "shuffle").type == "bool"


def test_from_yaml_learns_groups_fields_and_bounds(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
dataset:
  name: oxe
  batch_size: 32
training:
  learning_rate:
    type: float
    bounds:
      min: 1.0e-5
      max: 1.0e-2
  optimizer:
    type: str
    values: [adam, sgd]
"""
    )
    cfg = Config.from_yaml(yaml_path)

    assert set(cfg.groups()) == {"dataset", "training"}
    assert set(cfg.fields("training")) == {"learning_rate", "optimizer"}

    lr_spec = cfg.schema("training", "learning_rate")
    assert lr_spec.type == "float"
    assert lr_spec.bounds == {"min": 1e-5, "max": 1e-2}

    opt_spec = cfg.schema("training", "optimizer")
    assert opt_spec.values == ["adam", "sgd"]


def test_from_json_roundtrip(tmp_path):
    json_path = tmp_path / "config.json"
    data = {"dataset": {"name": "libero", "batch_size": 16}}
    json_path.write_text(json.dumps(data))

    cfg = Config.from_json(json_path)
    assert cfg.dataset.name == "libero"
    assert cfg.dataset.batch_size == 16


def test_save_and_reload_round_trips(tmp_path):
    cfg = Config()
    cfg.define("training", "learning_rate", default=1e-4, type="float")
    cfg.training(learning_rate={"type": "float", "bounds": {"min": 1e-5, "max": 1e-2}})
    out_path = tmp_path / "out.yaml"
    cfg.save(out_path)

    reloaded = Config.from_yaml(out_path)
    assert reloaded.schema("training", "learning_rate").bounds == {"min": 1e-5, "max": 1e-2}


def test_to_sweep_maps_bounds_values_and_fixed_fields():
    cfg = Config()
    cfg.define("training", "learning_rate", default=1e-4, type="float")
    cfg.define("training", "optimizer", default="adam", type="str")
    cfg.define("training", "num_epochs", default=100, type="int")
    cfg.training(
        learning_rate={"type": "float", "bounds": {"min": 1e-5, "max": 1e-2}},
        optimizer={"type": "str", "values": ["adam", "sgd"]},
        num_epochs=100,
    )

    sweep = cfg.to_sweep(method="bayes", metric={"name": "loss", "goal": "minimize"})

    assert sweep["method"] == "bayes"
    assert sweep["metric"] == {"name": "loss", "goal": "minimize"}
    assert sweep["parameters"]["training.learning_rate"] == {"min": 1e-5, "max": 1e-2}
    assert sweep["parameters"]["training.optimizer"] == {"values": ["adam", "sgd"]}
    assert sweep["parameters"]["training.num_epochs"] == {"value": 100}


def test_group_attribute_access_raises_for_unknown_field():
    cfg = Config()
    cfg.define("dataset", "name", default="oxe")
    cfg.dataset(name="oxe")
    with pytest.raises(AttributeError):
        cfg.dataset.missing_field


def test_set_bounds_raises_for_unknown_field():
    cfg = Config()
    cfg.define("training", "learning_rate", default=1e-4, type="float")
    cfg.training(learning_rate=1e-4)
    with pytest.raises(KeyError):
        cfg.set_bounds("training", "missing_field", min=0, max=1)
