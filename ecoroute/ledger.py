"""SQLite-backed request ledger, response cache and budget settings."""
from __future__ import annotations

import hashlib
import json
import re
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
    team TEXT,
    feedback INTEGER,              -- +1 thumbs up, -1 thumbs down
    feedback_comment TEXT,
    escalated INTEGER DEFAULT 0,   -- 1 = this answer replaced a weaker model's failed attempt
    check_ok INTEGER               -- answer check result when cascade is on (1 ok, 0 failed)
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, created REAL, model_id TEXT, response TEXT,
    input_tokens INTEGER, output_tokens INTEGER, hits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS payloads (request_id INTEGER PRIMARY KEY, created REAL, messages TEXT, response TEXT);
CREATE TABLE IF NOT EXISTS batch_jobs (
    id TEXT PRIMARY KEY, created REAL, deadline REAL, status TEXT, team TEXT, policy TEXT,
    max_carbon_g_per_kwh REAL, submit_g_per_kwh REAL, n INTEGER, done INTEGER DEFAULT 0, finished REAL
);
CREATE TABLE IF NOT EXISTS batch_items (
    job_id TEXT, idx INTEGER, custom_id TEXT, payload TEXT, status TEXT, request_id INTEGER,
    response TEXT, error TEXT, model_id TEXT, grid_g_per_kwh REAL, ran REAL, PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, request_id INTEGER, tier INTEGER, task_type TEXT,
    model_id TEXT, baseline_model TEXT, judge_model TEXT,
    score_routed REAL, score_baseline REAL, verdict TEXT, reason TEXT,
    overhead_energy_wh REAL, overhead_cost_usd REAL
);
"""
# Columns added after the first release; older databases are migrated on open.
MIGRATIONS = {"requests": {"team": "TEXT", "feedback": "INTEGER", "feedback_comment": "TEXT",
                            "escalated": "INTEGER DEFAULT 0", "check_ok": "INTEGER"}}


CONTRACTIONS = {"what's": "what is", "who's": "who is", "where's": "where is", "how's": "how is", "it's": "it is",
                "that's": "that is", "there's": "there is", "i'm": "i am", "don't": "do not", "doesn't": "does not",
                "can't": "cannot", "won't": "will not", "isn't": "is not", "aren't": "are not", "whats": "what is"}
FILLER = re.compile(r"^(?:(?:hey|hi|hello|please|pls|kindly|can you|could you|would you|will you|tell me|"
                    r"i want to know|i'd like to know|quick question)\b[\s,:]*)+|"
                    r"(?:[\s,]*\b(?:please|pls|thanks|thank you|thx)\b)+[\s.!?]*$")
ARTICLES = re.compile(r"\b(?:a|an|the)\b")
CODEISH = re.compile(r"[{}()\[\]=<>+*/^`;|\\]|\d\s*-\s*\d")


def canonical(text: str) -> str:
    """Canonical form of a prompt for the response cache.

    Plain-language prompts lose case, punctuation, articles and polite filler, so "What's the capital of
    France?" and "please tell me what is the capital of france" share one cache entry. Prompts that look like
    code or maths only lose case and extra whitespace, since a symbol there can change the meaning.
    """
    t = " ".join(str(text or "").lower().split())
    if CODEISH.search(t):
        return t
    t = t.replace("\u2019", "'")
    for k, v in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(k)}(?=\W|$)", v, t)
    t = re.sub(r"[^\w\s.%$#@&'-]", " ", t)       # drop ?, !, commas, quotes…
    t = re.sub(r"(?<!\d)\.(?!\d)", " ", t)          # …and full stops, but keep decimals like 3.5
    t = FILLER.sub("", " ".join(t.split()))
    t = ARTICLES.sub(" ", t)
    return " ".join(t.split())


def cache_key(messages: list[dict], tier_floor: int | None = None, variant: str = "") -> str:
    norm = [{"role": m.get("role"), "content": canonical(m.get("content", ""))} for m in messages]
    return hashlib.sha256(json.dumps([norm, tier_floor, variant], sort_keys=True).encode()).hexdigest()


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

    def update(self, request_id: int, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE requests SET {sets} WHERE id = ?", (*fields.values(), request_id))
            self._conn.commit()

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

    def get(self, request_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        return dict(r) if r else None

    def set_feedback(self, request_id: int, rating: int, comment: str | None = None) -> None:
        with self._lock:
            self._conn.execute("UPDATE requests SET feedback = ?, feedback_comment = ? WHERE id = ?",
                               (rating, comment, request_id))
            self._conn.commit()

    def feedback_stats(self, since: float) -> dict[str, dict]:
        """Per-model thumbs up/down counts, used to nudge routing away from models users dislike."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT model_id, SUM(feedback = 1) up, SUM(feedback = -1) down FROM requests
                   WHERE ts >= ? AND feedback IS NOT NULL GROUP BY model_id""", (since,)).fetchall()
        return {r["model_id"]: {"up": r["up"] or 0, "down": r["down"] or 0} for r in rows}

    # ----------------------------- payloads -----------------------------
    def payload_put(self, request_id: int, messages: list[dict], response: str, keep_s: float) -> None:
        """Keep the full prompt and answer for a while, so feedback can re-ask and evals can grade them."""
        with self._lock:
            self._conn.execute("DELETE FROM payloads WHERE created < ?", (time.time() - keep_s,))
            self._conn.execute("INSERT OR REPLACE INTO payloads VALUES (?, ?, ?, ?)",
                               (request_id, time.time(), json.dumps(messages), response))
            self._conn.commit()

    def payload_get(self, request_id: int) -> list[dict] | None:
        p = self.payload(request_id)
        return p["messages"] if p else None

    def payload(self, request_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT messages, response FROM payloads WHERE request_id = ?",
                                   (request_id,)).fetchone()
        return {"messages": json.loads(r["messages"]), "response": r["response"]} if r else None

    # ------------------------------ evals -------------------------------
    def eval_put(self, row: dict) -> int:
        cols = ", ".join(row)
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO evals ({cols}) VALUES ({', '.join('?' for _ in row)})",
                                     list(row.values()))
            self._conn.commit()
            return cur.lastrowid

    def quality(self, since: float = 0.0) -> dict:
        with self._lock:
            ev = dict(self._conn.execute(
                """SELECT COUNT(*) n, COALESCE(SUM(verdict = 'routed_better'), 0) routed_better,
                          COALESCE(SUM(verdict = 'equal'), 0) equal, COALESCE(SUM(verdict = 'baseline_better'), 0) baseline_better,
                          AVG(score_routed) avg_routed, AVG(score_baseline) avg_baseline,
                          COALESCE(SUM(overhead_energy_wh), 0) overhead_energy_wh,
                          COALESCE(SUM(overhead_cost_usd), 0) overhead_cost_usd
                   FROM evals WHERE ts >= ?""", (since,)).fetchone())
            ev_tier = [dict(x) for x in self._conn.execute(
                """SELECT tier, COUNT(*) n, SUM(verdict != 'baseline_better') at_parity FROM evals
                   WHERE ts >= ? GROUP BY tier ORDER BY tier""", (since,))]
            recent = [dict(x) for x in self._conn.execute(
                "SELECT * FROM evals WHERE ts >= ? ORDER BY id DESC LIMIT 20", (since,))]
            fb = dict(self._conn.execute(
                """SELECT COALESCE(SUM(feedback = 1), 0) up, COALESCE(SUM(feedback = -1), 0) down,
                          COALESCE(SUM(escalated), 0) escalated, COALESCE(SUM(check_ok = 0), 0) checks_failed,
                          COALESCE(SUM(check_ok IS NOT NULL), 0) checks_run
                   FROM requests WHERE ts >= ?""", (since,)).fetchone())
        ev["parity_rate"] = (ev["routed_better"] + ev["equal"]) / ev["n"] if ev["n"] else None
        rated = fb["up"] + fb["down"]
        fb["satisfaction"] = fb["up"] / rated if rated else None
        return {"evals": ev, "evals_by_tier": ev_tier, "recent_evals": recent, "feedback": fb}

    # --------------------------- reporting -----------------------------
    def period_by_model(self, start: float, end: float, team: str | None = None) -> list[dict]:
        where, targs = _team_filter(team)
        with self._lock:
            return [dict(x) for x in self._conn.execute(
                f"""SELECT model_id, provider, COUNT(*) n, SUM(source = 'cache') cache_hits,
                          SUM(input_tokens) input_tokens, SUM(output_tokens) output_tokens,
                          SUM(energy_wh) energy_wh, SUM(carbon_g) carbon_g, SUM(water_ml) water_ml, SUM(cost_usd) cost_usd,
                          SUM(baseline_energy_wh) baseline_energy_wh, SUM(baseline_carbon_g) baseline_carbon_g,
                          SUM(baseline_water_ml) baseline_water_ml, SUM(baseline_cost_usd) baseline_cost_usd
                   FROM requests WHERE ts >= ? AND ts < ?{where} GROUP BY model_id ORDER BY energy_wh DESC""",
                (start, end, *targs))]

    def period_by_team(self, start: float, end: float) -> list[dict]:
        with self._lock:
            return [dict(x) for x in self._conn.execute(
                """SELECT COALESCE(team, 'unassigned') team, COUNT(*) n, SUM(energy_wh) energy_wh,
                          SUM(carbon_g) carbon_g, SUM(water_ml) water_ml, SUM(cost_usd) cost_usd,
                          SUM(baseline_energy_wh) baseline_energy_wh, SUM(baseline_carbon_g) baseline_carbon_g,
                          SUM(baseline_cost_usd) baseline_cost_usd
                   FROM requests WHERE ts >= ? AND ts < ? GROUP BY COALESCE(team, 'unassigned') ORDER BY n DESC""",
                (start, end))]

    # ----------------------------- batches ------------------------------
    def batch_create(self, job: dict, items: list[dict]) -> None:
        with self._lock:
            self._conn.execute(f"INSERT INTO batch_jobs ({', '.join(job)}) VALUES ({', '.join('?' for _ in job)})",
                               list(job.values()))
            self._conn.executemany(
                "INSERT INTO batch_items (job_id, idx, custom_id, payload, status) VALUES (?, ?, ?, ?, 'queued')",
                [(job["id"], i, it.get("custom_id"), json.dumps(it)) for i, it in enumerate(items)])
            self._conn.commit()

    def batch_jobs(self, statuses: tuple[str, ...] = ("queued", "running")) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM batch_jobs WHERE status IN ({', '.join('?' for _ in statuses)}) ORDER BY deadline",
                statuses).fetchall()
        return [dict(r) for r in rows]

    def batch_get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._conn.execute("SELECT * FROM batch_jobs WHERE id = ?", (job_id,)).fetchone()
            if not job:
                return None
            items = self._conn.execute("SELECT * FROM batch_items WHERE job_id = ? ORDER BY idx", (job_id,)).fetchall()
        out = dict(job)
        out["items"] = [{**{k: v for k, v in dict(r).items() if k != "payload"}, "request": json.loads(r["payload"])}
                        for r in items]
        return out

    def batch_item_update(self, job_id: str, idx: int, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE batch_items SET {sets} WHERE job_id = ? AND idx = ?",
                               (*fields.values(), job_id, idx))
            self._conn.commit()

    def batch_update(self, job_id: str, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE batch_jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requests")
            self._conn.execute("DELETE FROM payloads")
            self._conn.execute("DELETE FROM evals")
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
