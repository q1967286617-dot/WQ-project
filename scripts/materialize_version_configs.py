from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.versioning import (
    DEFAULT_VERSION_REGISTRY,
    load_version_specs,
    materialize_version_configs,
)
from src.utils.paths import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize versioned model/backtest configs from the central registry.")
    parser.add_argument("--version_registry", default=str(DEFAULT_VERSION_REGISTRY))
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    parser.add_argument("--out_dir", default="configs/versions")
    parser.add_argument("--versions", nargs="*", default=None)
    args = parser.parse_args()

    registry_path = PROJECT_ROOT / args.version_registry
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model_cfg = load_yaml(PROJECT_ROOT / args.model_cfg)
    base_backtest_cfg = load_yaml(PROJECT_ROOT / args.backtest_cfg)
    specs = load_version_specs(registry_path)
    versions = args.versions or list(specs.keys())

    manifest: dict[str, dict] = {}
    for version in versions:
        if version not in specs:
            raise SystemExit(f"Unknown version: {version}")
        spec = materialize_version_configs(
            version=version,
            base_model_cfg=base_model_cfg,
            base_backtest_cfg=base_backtest_cfg,
            out_model_path=out_dir / f"{version}_model.yaml",
            out_backtest_path=out_dir / f"{version}_backtest.yaml",
            registry_path=registry_path,
        )
        manifest[version] = spec.to_manifest()

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
