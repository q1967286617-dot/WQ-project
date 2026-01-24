from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def is_kaggle_env() -> bool:
    """Best-effort detection for Kaggle Notebook runtime."""
    return Path("/kaggle/input").exists() and Path("/kaggle/working").exists()


@dataclass(frozen=True)
class ProjectPaths:
    tableA_path: Path
    tableB_path: Path
    processed_dir: Path
    models_dir: Path
    outputs_dir: Path

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(paths_cfg: Dict[str, Any], project_root: Optional[Path] = None) -> ProjectPaths:
    """
    Resolve relative paths against project_root (defaults to repo root),
    and apply Kaggle read-only/output redirection if needed.
    """
    root = project_root or Path.cwd()

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else (root / p)

    tableA_path = _p(paths_cfg["tableA_path"])
    tableB_path = _p(paths_cfg.get("tableB_path", ""))
    processed_dir = _p(paths_cfg["processed_dir"])
    models_dir = _p(paths_cfg["models_dir"])
    outputs_dir = _p(paths_cfg["outputs_dir"])

    if is_kaggle_env():
        # If users mount datasets as /kaggle/input/<name>/..., paths in yaml may be overwritten in notebooks.
        # Here we only ensure writable outputs go to /kaggle/working.
        outputs_dir = Path("/kaggle/working") / "outputs" / "runs"
        processed_dir = Path("/kaggle/working") / "data" / "processed"
        models_dir = Path("/kaggle/working") / "models"

    return ProjectPaths(
        tableA_path=tableA_path,
        tableB_path=tableB_path,
        processed_dir=processed_dir,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
