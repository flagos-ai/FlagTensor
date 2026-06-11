from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import yaml


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: str
    python_api: str
    impl_file: str
    correctness_test: str
    benchmark_test: str
    correctness_mark: str
    benchmark_mark: str
    benchmark_modes: Tuple[str, ...]
    status: str
    skip_reason: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status.lower() != "disabled"

    @property
    def is_blocked(self) -> bool:
        return self.status.lower() == "blocked"

    def supports_mode(self, mode: str) -> bool:
        return mode in self.benchmark_modes


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "conf" / "operators.yaml"


def load_operator_registry(registry_path: Optional[Union[str, Path]] = None) -> List[OperatorSpec]:
    path = Path(registry_path) if registry_path else _default_registry_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = payload.get("ops", [])
    specs = []
    for item in items:
        specs.append(
            OperatorSpec(
                name=item["name"],
                category=item["category"],
                python_api=item["python_api"],
                impl_file=item["impl_file"],
                correctness_test=item["correctness_test"],
                benchmark_test=item["benchmark_test"],
                correctness_mark=item.get("correctness_mark", item["name"]),
                benchmark_mark=item.get("benchmark_mark", item["name"]),
                benchmark_modes=tuple(item.get("benchmark_modes", ["operator"])),
                status=item.get("status", "stable"),
                skip_reason=item.get("skip_reason"),
            )
        )
    return specs


def get_operator_map(registry_path: Optional[Union[str, Path]] = None) -> Dict[str, OperatorSpec]:
    return {spec.name: spec for spec in load_operator_registry(registry_path=registry_path)}


def filter_operator_specs(
    specs: Iterable[OperatorSpec],
    *,
    names: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None,
    mode: Optional[str] = None,
    include_blocked: bool = False,
) -> List[OperatorSpec]:
    selected_names = {name.strip().lower() for name in names or [] if name and name.strip()}
    excluded_names = {name.strip().lower() for name in exclude_names or [] if name and name.strip()}
    selected_categories = {item.strip().lower() for item in categories or [] if item and item.strip()}
    filtered = []
    for spec in specs:
        if selected_names and spec.name.lower() not in selected_names:
            continue
        if spec.name.lower() in excluded_names:
            continue
        if selected_categories and spec.category.lower() not in selected_categories:
            continue
        if mode and not spec.supports_mode(mode):
            continue
        if spec.is_blocked and not include_blocked:
            continue
        if not spec.is_active:
            continue
        filtered.append(spec)
    return filtered
