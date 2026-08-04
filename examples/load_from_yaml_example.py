"""Example: Config.from_yaml() alone is enough for CLI overrides to work.

Loading a file (from_yaml/from_json/from_file) automatically syncs the
config's internal argparse parser with every field it just learned -- no
add_argument()/add_arguments() call needed before parse_args() works.

    python examples/load_from_yaml_example.py
    python examples/load_from_yaml_example.py --training.learning_rate 5e-4 --dataset.batch_size 64
    python examples/load_from_yaml_example.py --help
"""

from pathlib import Path

from training_cfgs import Config

CONFIG_PATH = Path(__file__).parent / "sample_config.yaml"


def main() -> None:
    cfg = Config.from_yaml(CONFIG_PATH)   # parser is already synced at this point
    cfg.parse_args()                      # parses sys.argv against cfg.parser, applies overrides
    print(cfg)


if __name__ == "__main__":
    main()
