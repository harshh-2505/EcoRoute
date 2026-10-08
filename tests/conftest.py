import pytest

from ecoroute.config import Settings
from ecoroute.service import EcoRouteService


@pytest.fixture
def settings(tmp_path):
    s = Settings.from_env()
    s.mode = "simulate"
    s.db_path = str(tmp_path / "test.db")
    s.api_keys = {}
    s.monthly_budget_usd = 0
    s.monthly_energy_budget_wh = 0
    return s


@pytest.fixture
def service(settings):
    svc = EcoRouteService(settings)
    svc.simulator.delay = False
    yield svc
    svc.ledger.close()
