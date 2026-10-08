"""Model catalog and grid-region assumptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ModelSpec:
    id: str
    provider: str
    model: str
    display_name: str
    kind: str  # "hosted" | "local"
    price_in: float
    price_out: float
    quality: float
    max_tier: int
    tps: float
    region: str
    pue: float
    wue: float
    params_total_b: float | None = None
    params_active_b: float | None = None
    device_watts: float | None = None

    @property
    def is_local(self) -> bool:
        return self.kind == "local"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Region:
    key: str
    name: str
    carbon_g_per_kwh: float
    water_l_per_kwh: float
    zone: str | None = None
    live: bool = False  # True when carbon intensity came from a live API


class Catalog:
    def __init__(self, models: list[ModelSpec], regions: dict[str, Region], local_region: str,
                 pricing_snapshot: str = ""):
        self.models = models
        self.regions = regions
        self.local_region = local_region
        self.pricing_snapshot = pricing_snapshot
        self._by_id = {m.id: m for m in models}

    @classmethod
    def load(cls, catalog_path: str | Path, grid_path: str | Path,
             local_region: str | None = None) -> Catalog:
        raw = yaml.safe_load(Path(catalog_path).read_text())
        grid = yaml.safe_load(Path(grid_path).read_text())
        fields = set(ModelSpec.__dataclass_fields__)
        models = [ModelSpec(**{k: v for k, v in m.items() if k in fields}) for m in raw["models"]]
        regions = {k: Region(key=k, **v) for k, v in grid["regions"].items()}
        lr = local_region or grid.get("local_region", "global")
        if lr not in regions:
            raise ValueError(f"Unknown local region '{lr}'. Options: {sorted(regions)}")
        ids = [m.id for m in models]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate model ids in catalog")
        for m in models:
            if m.region != "local" and m.region not in regions:
                raise ValueError(f"Model {m.id} uses unknown region '{m.region}'")
            if not 1 <= m.max_tier <= 5:
                raise ValueError(f"Model {m.id}: max_tier must be 1-5")
            if m.is_local and not m.device_watts:
                raise ValueError(f"Local model {m.id} needs device_watts")
        return cls(models, regions, lr, raw.get("pricing_snapshot", ""))

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise KeyError(f"Unknown model id '{model_id}'") from None

    def has(self, model_id: str) -> bool:
        return model_id in self._by_id

    def region_for(self, model: ModelSpec) -> Region:
        return self.regions[self.local_region if model.region == "local" else model.region]
