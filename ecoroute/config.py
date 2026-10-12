"""Runtime configuration, read from environment variables (optionally a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # "simulate": never call real APIs; "live": only route to providers with credentials;
    # "auto": live if any provider is configured, otherwise simulate.
    mode: str = "auto"
    db_path: str = "ecoroute.db"
    catalog_path: str = str(DATA_DIR / "catalog.yaml")
    grid_path: str = str(DATA_DIR / "grid.yaml")
    default_policy: str = "balanced"
    baseline_model: str = "openai-gpt-5"  # what you'd use if you always picked the flagship
    local_region: str | None = None
    cache_enabled: bool = True
    cache_ttl_hours: float = 24.0
    store_prompts: bool = True
    monthly_budget_usd: float = 0.0  # 0 = no budget
    monthly_energy_budget_wh: float = 0.0  # 0 = no budget
    request_timeout_s: float = 60.0
    ollama_host: str = "http://localhost:11434"
    electricitymaps_api_key: str | None = None
    api_keys: dict[str, str | None] = field(default_factory=dict)
    # Client access keys → team name. Empty = open access (fine for local use and demos).
    access_keys: dict[str, str | None] = field(default_factory=dict)
    admin_key: str | None = None  # required for reset/budget/seed when set
    cascade: bool = False         # check each answer; re-ask a stronger model if it clearly failed
    shadow_rate: float = 0.0      # share of live requests also sent to the baseline and graded by a judge
    judge_model: str | None = None  # catalog id of the judge (default: the baseline model)

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        return cls(
            mode=os.getenv("ECOROUTE_MODE", "auto").lower(),
            db_path=os.getenv("ECOROUTE_DB", "ecoroute.db"),
            catalog_path=os.getenv("ECOROUTE_CATALOG", str(DATA_DIR / "catalog.yaml")),
            grid_path=os.getenv("ECOROUTE_GRID", str(DATA_DIR / "grid.yaml")),
            default_policy=os.getenv("ECOROUTE_POLICY", "balanced"),
            baseline_model=os.getenv("ECOROUTE_BASELINE_MODEL", "openai-gpt-5"),
            local_region=os.getenv("ECOROUTE_LOCAL_REGION") or None,
            cache_enabled=_bool("ECOROUTE_CACHE", True),
            cache_ttl_hours=_float("ECOROUTE_CACHE_TTL_HOURS", 24.0),
            store_prompts=_bool("ECOROUTE_STORE_PROMPTS", True),
            monthly_budget_usd=_float("ECOROUTE_MONTHLY_BUDGET_USD", 0.0),
            monthly_energy_budget_wh=_float("ECOROUTE_MONTHLY_ENERGY_BUDGET_WH", 0.0),
            request_timeout_s=_float("ECOROUTE_TIMEOUT_S", 60.0),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            electricitymaps_api_key=os.getenv("ELECTRICITYMAPS_API_KEY") or None,
            access_keys=parse_access_keys(os.getenv("ECOROUTE_API_KEYS", "")),
            admin_key=os.getenv("ECOROUTE_ADMIN_KEY") or None,
            cascade=_bool("ECOROUTE_CASCADE", False),
            shadow_rate=min(1.0, max(0.0, _float("ECOROUTE_SHADOW_RATE", 0.0))),
            judge_model=os.getenv("ECOROUTE_JUDGE_MODEL") or None,
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY") or None,
                "anthropic": os.getenv("ANTHROPIC_API_KEY") or None,
                "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
                "groq": os.getenv("GROQ_API_KEY") or None,
                "mistral": os.getenv("MISTRAL_API_KEY") or None,
                "deepseek": os.getenv("DEEPSEEK_API_KEY") or None,
                "xai": os.getenv("XAI_API_KEY") or None,
            },
        )


def parse_access_keys(raw: str) -> dict[str, str | None]:
    """``"key1:team-a, key2:team-b, key3"`` → ``{"key1": "team-a", "key2": "team-b", "key3": None}``."""
    out: dict[str, str | None] = {}
    for item in raw.split(","):
        key, _, team = item.strip().partition(":")
        if key.strip():
            out[key.strip()] = team.strip() or None
    return out
