"""SQLite-backed request ledger, response cache and budget settings."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    prompt_preview TEXT,
    complexity REAL, tier INTEGER, task_type TEXT, policy TEXT,
    model_id TEXT, provider TEXT, source TEXT,          -- source: live | simulated | cache
    input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL,
    energy_wh REAL, carbon_g REAL, water_ml REAL, cost_usd REAL,
    baseline_model TEXT,
    baseline_energy_wh REAL, baseline_carbon_g REAL, baseline_water_ml REAL, baseline_cost_usd REAL,
    sustainability_score REAL,
    fallback_used INTEGER DEFAULT 0,
    team TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, created REAL, model_id TEXT, response TEXT,
    input_tokens INTEGER, output_tokens INTEGER, hits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""
# Columns added after the first release; older databases are migrated on open.
MIGRATIONS = {"requests": {"team": "TEXT"}}


def cache_key(messages: list[dict], tier_floor: int | None = None) -> str:
    norm = [{"role": m.get("role"), "content": " ".join(str(m.get("content", "")).lower().split())}
            for m in messages]
    return hashlib.sha256(json.dumps([norm, tier_floor], sort_keys=True).encode()).hexdigest()


def _team_filter(team: str | None) -> tuple[str, tuple]:
    if not team:
        return "", ()
    if team == "unassigned":
        return " AND team IS NULL", ()
    return " AND team = ?", (team,)


def month_start_ts(now: float | None = None) -> float:
    d = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


class Ledger:
    def __init__(self, path: str = "ecoroute.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                for col, typ in cols.items():
                    if col not in have:
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ----------------------------- requests -----------------------------
    def record(self, row: dict) -> int:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO requests ({cols}) VALUES ({qs})", list(row.values()))
            self._conn.commit()
            return cur.lastrowid

    def set_ts(self, request_id: int, ts: float) -> None:
        with self._lock:
            self._conn.execute("UPDATE requests SET ts = ? WHERE id = ?", (ts, request_id))
            self._conn.commit()

    def recent(self, limit: int = 50, team: str | None = None) -> list[dict]:
        where, args = _team_filter(team)
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM requests WHERE 1=1{where} ORDER BY id DESC LIMIT ?",
                                      (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requests")
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    def summary(self, since: float = 0.0, team: str | None = None) -> dict:
        where, targs = _team_filter(team)
        args = (since, *targs)
        q = """SELECT COUNT(*) n,
                 COALESCE(SUM(energy_wh),0) energy_wh, COALESCE(SUM(carbon_g),0) carbon_g,
                 COALESCE(SUM(water_ml),0) water_ml, COALESCE(SUM(cost_usd),0) cost_usd,
                 COALESCE(SUM(baseline_energy_wh),0) b_energy, COALESCE(SUM(baseline_carbon_g),0) b_carbon,
                 COALESCE(SUM(baseline_water_ml),0) b_water, COALESCE(SUM(baseline_cost_usd),0) b_cost,
                 COALESCE(SUM(input_tokens),0) in_tok, COALESCE(SUM(output_tokens),0) out_tok,
                 COALESCE(AVG(latency_ms),0) avg_latency, COALESCE(AVG(sustainability_score),0) avg_sust,
                 COALESCE(SUM(source='cache'),0) cache_hits, COALESCE(SUM(source='live'),0) live,
                 COALESCE(SUM(fallback_used),0) fallbacks
               FROM requests WHERE ts >= ?{where}"""
        with self._lock:
            r = dict(self._conn.execute(q.format(where=where), args).fetchone())
            by_model = [dict(x) for x in self._conn.execute(
                f"""SELECT model_id, provider, COUNT(*) n, SUM(energy_wh) energy_wh, SUM(carbon_g) carbon_g,
                          SUM(water_ml) water_ml, SUM(cost_usd) cost_usd, AVG(latency_ms) avg_latency
                   FROM requests WHERE ts >= ?{where} GROUP BY model_id ORDER BY n DESC""", args)]
            by_tier = [dict(x) for x in self._conn.execute(
                f"""SELECT tier, COUNT(*) n, SUM(energy_wh) energy_wh, SUM(baseline_energy_wh) baseline_energy_wh,
                          SUM(cost_usd) cost_usd FROM requests WHERE ts >= ?{where} GROUP BY tier ORDER BY tier""",
                args)]
            by_task = [dict(x) for x in self._conn.execute(
                f"SELECT task_type, COUNT(*) n FROM requests WHERE ts >= ?{where} GROUP BY task_type ORDER BY n DESC",
                args)]
            by_team = [dict(x) for x in self._conn.execute(
                f"""SELECT COALESCE(team, 'unassigned') team, COUNT(*) n, SUM(energy_wh) energy_wh,
                          SUM(baseline_energy_wh) baseline_energy_wh, SUM(carbon_g) carbon_g, SUM(cost_usd) cost_usd,
                          SUM(baseline_cost_usd) baseline_cost_usd
                   FROM requests WHERE ts >= ?{where} GROUP BY COALESCE(team, 'unassigned') ORDER BY n DESC""",
                args)]
        tok = r["in_tok"] + r["out_tok"]
        r["energy_per_1k_tokens_wh"] = r["energy_wh"] / tok * 1000 if tok else 0.0
        r["cost_per_request_usd"] = r["cost_usd"] / r["n"] if r["n"] else 0.0
        r["cache_hit_rate"] = r["cache_hits"] / r["n"] if r["n"] else 0.0
        return {"totals": r, "by_model": by_model, "by_tier": by_tier, "by_task": by_task, "by_team": by_team}

    def timeseries(self, since: float, bucket_s: int, team: str | None = None) -> list[dict]:
        where, targs = _team_filter(team)
        q = """SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, COUNT(*) n,
                      SUM(energy_wh) energy_wh, SUM(baseline_energy_wh) baseline_energy_wh,
                      SUM(carbon_g) carbon_g, SUM(baseline_carbon_g) baseline_carbon_g,
                      SUM(water_ml) water_ml, SUM(cost_usd) cost_usd, SUM(baseline_cost_usd) baseline_cost_usd
               FROM requests WHERE ts >= ?{where} GROUP BY bucket ORDER BY bucket"""
        with self._lock:
            return [dict(r) for r in self._conn.execute(q.format(where=where), (bucket_s, bucket_s, since, *targs))]

    def month_to_date(self) -> dict:
        with self._lock:
            r = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(energy_wh),0) e FROM requests WHERE ts >= ?",
                (month_start_ts(),)).fetchone()
        return {"cost_usd": r["c"], "energy_wh": r["e"]}

    # ------------------------------ cache -------------------------------
    def cache_get(self, key: str, ttl_s: float) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM cache WHERE key = ?", (key,)).fetchone()
            if not r:
                return None
            if time.time() - r["created"] > ttl_s:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            self._conn.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            self._conn.commit()
            return dict(r)

    def cache_put(self, key: str, model_id: str, response: str, in_tok: int, out_tok: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, created, model_id, response, input_tokens, output_tokens, hits)"
                " VALUES (?, ?, ?, ?, ?, ?, 0)", (key, time.time(), model_id, response, in_tok, out_tok))
            self._conn.commit()

    # ----------------------------- settings -----------------------------
    def get_setting(self, key: str, default=None):
        with self._lock:
            r = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_setting(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                               (key, json.dumps(value)))
            self._conn.commit()
