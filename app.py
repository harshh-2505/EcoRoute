"""Vercel entrypoint: exposes the EcoRoute FastAPI app as `app`."""
import os

# Vercel's filesystem is read-only except /tmp, so the SQLite ledger lives there (it resets on cold starts).
if os.getenv("VERCEL"):
    os.environ.setdefault("ECOROUTE_DB", "/tmp/ecoroute.db")

from ecoroute.api import create_app

app = create_app()
