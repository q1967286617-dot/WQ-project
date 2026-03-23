from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.paths import load_yaml


DEFAULT_VERSION_REGISTRY = Path("configs/version_registry.yaml")


@dataclass(frozen=True)
class VersionSpec:
    version: str
    label: str
    description: str
    num_cols: list[str]
    cat_cols: list[str]
    num_boost_round: int
    early_stopping_rounds: int
    dividend_rules_mode: str
    runner_kind: str = "xgb"
    random_seed: int = 42
    report_targets: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["n_num"] = len(self.num_cols)
        payload["n_cat"] = len(self.cat_cols)
        return payload


def _resolve_registry_path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else DEFAULT_VERSION_REGISTRY


def _expand_groups(group_names: list[str], feature_groups: dict[str, list[str]]) -> list[str]:
    expanded: list[str] = []
    for name in group_names:
        if name not in feature_groups:
            raise KeyError(f"Unknown feature group: {name}")
        expanded.extend(feature_groups[name])
    return list(dict.fromkeys(expanded))


def _registry_payload(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = _resolve_registry_path(path)
    return load_yaml(registry_path)


def load_version_specs(path: str | Path | None = None) -> dict[str, VersionSpec]:
    payload = _registry_payload(path)
    feature_groups = payload.get("feature_groups", {})
    report_targets = payload.get("report_targets", {})
    versions = payload.get("versions", {})

    specs: dict[str, VersionSpec] = {}
    for version, item in versions.items():
        num_cols = item.get("num_cols")
        cat_cols = item.get("cat_cols")
        if num_cols is None:
            num_cols = _expand_groups(item.get("num_groups", []), feature_groups)
        if cat_cols is None:
            cat_cols = _expand_groups(item.get("cat_groups", []), feature_groups)

        specs[version] = VersionSpec(
            version=version,
            label=str(item.get("label", version)),
            description=str(item["description"]),
            num_cols=list(num_cols),
            cat_cols=list(cat_cols),
            num_boost_round=int(item.get("num_boost_round", 200)),
            early_stopping_rounds=int(item.get("early_stopping_rounds", 20)),
            dividend_rules_mode=str(item.get("dividend_rules_mode", "auto")),
            runner_kind=str(item.get("runner_kind", "xgb")),
            random_seed=int(item.get("random_seed", 42)),
            report_targets=dict(report_targets.get(version, {})),
        )
    return specs


def get_report_targets(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    payload = _registry_payload(path)
    return {str(k): dict(v) for k, v in payload.get("report_targets", {}).items()}


def get_version_label_map(path: str | Path | None = None) -> dict[str, str]:
    return {version: spec.label for version, spec in load_version_specs(path).items()}


def build_model_cfg(base_cfg: dict[str, Any], spec: VersionSpec) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["num_cols"] = list(spec.num_cols)
    cfg["cat_cols"] = list(spec.cat_cols)
    cfg["runner_kind"] = spec.runner_kind
    if spec.runner_kind == "random":
        cfg["random_seed"] = spec.random_seed
    cfg.setdefault("hyperparams", {})
    cfg["hyperparams"] = dict(cfg["hyperparams"])
    cfg["hyperparams"]["num_boost_round"] = spec.num_boost_round
    cfg["hyperparams"]["early_stopping_rounds"] = spec.early_stopping_rounds
    return cfg


def build_backtest_cfg(base_cfg: dict[str, Any], spec: VersionSpec) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["dividend_rules_mode"] = spec.dividend_rules_mode
    cfg["random_seed"] = spec.random_seed
    return cfg


def materialize_version_configs(
    version: str,
    base_model_cfg: dict[str, Any],
    base_backtest_cfg: dict[str, Any],
    out_model_path: Path,
    out_backtest_path: Path,
    registry_path: str | Path | None = None,
) -> VersionSpec:
    specs = load_version_specs(registry_path)
    if version not in specs:
        raise KeyError(f"Unknown version: {version}")
    spec = specs[version]

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_backtest_path.parent.mkdir(parents=True, exist_ok=True)
    with out_model_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(build_model_cfg(base_model_cfg, spec), f, allow_unicode=True, sort_keys=False)
    with out_backtest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(build_backtest_cfg(base_backtest_cfg, spec), f, allow_unicode=True, sort_keys=False)
    return spec
