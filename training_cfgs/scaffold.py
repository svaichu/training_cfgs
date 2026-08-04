"""Copy the bundled template config to a destination path."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def scaffold_config(dest: PathLike = "config.yaml") -> Path:
    """Copy the bundled, fully-commented template config to `dest`."""
    dest = Path(dest)
    with resources.as_file(resources.files("training_cfgs") / "template_config.yaml") as src:
        shutil.copy(src, dest)
    return dest
