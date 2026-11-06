"""Atomic, immutable closed events. Legacy rows remain readable and unchanged."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from contextlib import contextmanager
from uuid import UUID, uuid4, uuid5


DEFAULT_DB = Path(__file__).resolve().parents[2] / "data/alerts.db"


class StoreUnavailable(RuntimeError):
    """The operation did not commit. The caller can retry."""


class EventConflict(ValueError):
    """An immutable identity was reused with different content."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class EventStore:
    def __init__(self, path=None):
        self.path = str(path if path is not None else os.getenv("MICROSAT_EVENTS_DB", DEFAULT_DB))
        self._lock = threading.RLock()
        self.migration_backup = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False,
                                               isolation_level=None)
            self.connection.execute("PRAGMA busy_timeout=10000")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self._migrate()
        except sqlite3.Error as exc:
            raise StoreUnavailable(str(exc)) from exc

    @contextmanager
    def _transaction(self):
        with self._lock:
            began = False
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                began = True
                yield
                self.connection.execute("COMMIT")
            except Exception as exc:
                if began:
                    self.connection.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    raise StoreUnavailable(str(exc)) from exc
                raise

    def _migrate(self):
        tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        legacy = tables.intersection({"alerts", "segment_verdicts"})
        if legacy and "stream_events" not in tables and self.path != ":memory:":
            self.migration_backup = Path(self.path).with_name(Path(self.path).name + f".pre-events-v1-{uuid4()}.sqlite")
            with sqlite3.connect(str(self.migration_backup)) as backup:
                self.connection.backup(backup)
        statements = [
            "CREATE TABLE IF NOT EXISTS stream_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS stream_sessions (session_id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
            """CREATE TABLE IF NOT EXISTS stream_events (
                event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES stream_sessions(session_id),
                channel TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, content_sha TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS stream_alerts (
                event_id TEXT PRIMARY KEY REFERENCES stream_events(event_id), ts REAL NOT NULL,
                channel TEXT NOT NULL, severity TEXT, payload TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS stream_receipts (
                session_id TEXT NOT NULL REFERENCES stream_sessions(session_id), batch_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(session_id, batch_id))""",
            """CREATE TABLE IF NOT EXISTS legacy_event_map (
                legacy_table TEXT NOT NULL, legacy_id INTEGER NOT NULL, event_id TEXT UNIQUE NOT NULL,
                PRIMARY KEY(legacy_table, legacy_id))""",
            "CREATE INDEX IF NOT EXISTS stream_alert_ts ON stream_alerts(ts)",
            """CREATE TABLE IF NOT EXISTS stream_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT UNIQUE NOT NULL,
                event_id TEXT NOT NULL REFERENCES stream_events(event_id), status TEXT NOT NULL,
                created_at REAL NOT NULL, lease_until REAL NOT NULL, finished_at REAL,
                input TEXT NOT NULL, result TEXT)""",
            "CREATE INDEX IF NOT EXISTS stream_explanation_event ON stream_explanations(event_id,id)",
        ]
        with self._transaction():
            for sql in statements:
                self.connection.execute(sql)
            self.connection.execute("INSERT OR IGNORE INTO stream_meta VALUES ('db_uuid',?)", (str(uuid4()),))
            namespace = UUID(self.connection.execute("SELECT value FROM stream_meta WHERE key='db_uuid'").fetchone()[0])
            for table in sorted(legacy):
                for (identity,) in self.connection.execute(f"SELECT id FROM {table}").fetchall():
                    self.connection.execute("INSERT OR IGNORE INTO legacy_event_map VALUES (?,?,?)",
                        (table, identity, str(uuid5(namespace, f"{table}:{identity}"))))
            self.connection.execute("INSERT OR REPLACE INTO stream_meta VALUES ('schema_version','2')")

    def _save_session(self, session):
        previous = self.connection.execute("SELECT payload FROM stream_sessions WHERE session_id=?",
                                            (session["session_id"],)).fetchone()
        if previous:
            old = json.loads(previous[0])
            if any(old.get(key) != session.get(key) for key in ("source_id", "kind", "started_at")):
                raise EventConflict("session identity has different content")
        self.connection.execute("INSERT INTO stream_sessions VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload",
                                (session["session_id"], _json(session)))

    def save_session(self, session):
        _json(session)
        with self._transaction():
            self._save_session(session)

    def save_events(self, events, *, receipt=None, session=None):
        # Serialize all input before opening the transaction. NaN must never reach SQLite.
        encoded = [(event, _json(event)) for event in events]
        if receipt is not None:
            response = _json(receipt["response"])
        inserted = 0
        with self._transaction():
            if session is not None:
                self._save_session(session)
            for event, payload in encoded:
                digest = hashlib.sha256(payload.encode()).hexdigest()
                previous = self.connection.execute("SELECT content_sha FROM stream_events WHERE event_id=?",
                                                    (event["event_id"],)).fetchone()
                if previous:
                    if previous[0] != digest:
                        raise EventConflict("event_id has different content")
                    continue
                self.connection.execute("INSERT INTO stream_events VALUES (?,?,?,?,?,?)",
                    (event["event_id"], event["session_id"], event["channel"], event["status"], payload, digest))
                if event.get("alert") is not None:
                    alert = {**event["alert"], "event_id": event["event_id"],
                             "session_id": event["session_id"], "source_id": event["source_id"],
                             "ts": event["last_ts"], "status": event["status"]}
                    self.connection.execute("INSERT INTO stream_alerts VALUES (?,?,?,?,?)",
                        (event["event_id"], event["last_ts"], event["channel"], alert.get("severity"), _json(alert)))
                inserted += 1
            if receipt is not None:
                previous = self.connection.execute("SELECT fingerprint, response FROM stream_receipts WHERE session_id=? AND batch_id=?",
                    (receipt["session_id"], receipt["batch_id"])).fetchone()
                if previous and previous != (receipt["fingerprint"], response):
                    raise EventConflict("batch_id has different content")
                if previous is None:
                    self.connection.execute("INSERT INTO stream_receipts VALUES (?,?,?,?)",
                        (receipt["session_id"], receipt["batch_id"], receipt["fingerprint"], response))
        return inserted

    def _rows(self, sql, params=()):
        with self._lock:
            try:
                cursor = self.connection.execute(sql, params)
                names = [item[0] for item in cursor.description]
                return [dict(zip(names, row)) for row in cursor.fetchall()]
            except sqlite3.Error as exc:
                raise StoreUnavailable(str(exc)) from exc

    def get_session(self, session_id):
        rows = self._rows("SELECT payload FROM stream_sessions WHERE session_id=?", (session_id,))
        return json.loads(rows[0]["payload"]) if rows else None

    def get_receipt(self, session_id, batch_id):
        rows = self._rows("SELECT fingerprint,response FROM stream_receipts WHERE session_id=? AND batch_id=?",
                          (session_id, batch_id))
        return (rows[0]["fingerprint"], json.loads(rows[0]["response"])) if rows else None

    def _legacy(self, mapping):
        table = mapping["legacy_table"]
        row = self._rows(f"SELECT * FROM {table} WHERE id=?", (mapping["legacy_id"],))[0]
        raw = row.get("payload", row.get("features"))
        try:
            payload = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            payload = None
        return {**row, "event_id": mapping["event_id"], "status": "legacy_unverified",
                "legacy_table": table, "raw_payload": raw, "payload": payload,
                "waveform": None, "summary": payload.get("summary", "") if isinstance(payload, dict) else ""}

    def get_event(self, event_id):
        rows = self._rows("SELECT payload FROM stream_events WHERE event_id=?", (event_id,))
        if rows:
            return json.loads(rows[0]["payload"])
        mappings = self._rows("SELECT * FROM legacy_event_map WHERE event_id=?", (event_id,))
        return self._legacy(mappings[0]) if mappings else None

    def query_alerts(self, *, since=None, until=None, channel=None, severity=None, limit=100, offset=0):
        # Legacy and new identities share one stable time/order index for pagination.
        sources = "SELECT event_id, ts, channel, severity FROM stream_alerts"
        tables = self._rows("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
        if tables:
            sources += " UNION ALL SELECT m.event_id,a.ts,a.channel,a.severity FROM alerts a JOIN legacy_event_map m ON m.legacy_table='alerts' AND m.legacy_id=a.id"
        clauses, params = [], []
        for field, op, value in (("ts", ">=", since), ("ts", "<=", until),
                                 ("channel", "=", channel), ("severity", "=", severity)):
            if value is not None:
                clauses.append(f"{field} {op} ?")
                params.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"WITH all_alerts AS ({sources}) SELECT * FROM all_alerts{where}"
        total = self._rows(f"SELECT count(*) AS n FROM ({query})", params)[0]["n"]
        rows = self._rows(query + " ORDER BY ts DESC,event_id DESC LIMIT ? OFFSET ?",
                          [*params, max(1, min(int(limit), 1000)), max(0, int(offset))])
        items = []
        for row in rows:
            new = self._rows("SELECT payload FROM stream_alerts WHERE event_id=?", (row["event_id"],))
            items.append(json.loads(new[0]["payload"]) if new else self.get_event(row["event_id"]))
        return items, total

    def stats(self):
        total = self.query_alerts(limit=1)[1]
        # Group in SQL so the counts do not depend on page size.
        sources = "SELECT channel,severity FROM stream_alerts"
        if self._rows("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"):
            sources += " UNION ALL SELECT channel,severity FROM alerts"
        groups = {}
        for field in ("channel", "severity"):
            groups[f"by_{field}"] = {str(row[field]): row["n"] for row in self._rows(
                f"SELECT {field},count(*) AS n FROM ({sources}) GROUP BY {field}")}
        verdicts = self._rows("SELECT count(*) AS n FROM stream_events WHERE status='judged'")[0]["n"]
        if self._rows("SELECT name FROM sqlite_master WHERE type='table' AND name='segment_verdicts'"):
            verdicts += self._rows("SELECT count(*) AS n FROM segment_verdicts")[0]["n"]
        return {"alerts": total, "verdicts": verdicts, **groups}

    def close(self):
        with self._lock:
            self.connection.close()

    def _explanation_record(self, row):
        record = {**row, "input": json.loads(row["input"]),
                  "result": json.loads(row["result"]) if row["result"] else None}
        record.pop("id")
        if record["status"] == "generating" and record["lease_until"] < time.time():
            record["status"] = "failed"
            record["result"] = {"status": "generation_failed", "metadata": {"error": "generation_interrupted"}}
        return record

    def get_explanation(self, event_id):
        rows = self._rows("SELECT * FROM stream_explanations WHERE event_id=? ORDER BY id DESC LIMIT 1", (event_id,))
        return self._explanation_record(rows[0]) if rows else None

    def explanation_history(self, event_id):
        return [self._explanation_record(row) for row in self._rows(
            "SELECT * FROM stream_explanations WHERE event_id=? ORDER BY id", (event_id,))]

    def claim_explanation(self, event_id, inputs, *, retry=False, lease_seconds=300):
        payload = _json(inputs)
        with self._transaction():
            previous = self.get_explanation(event_id)
            if previous is not None and (previous["status"] == "generating" or not retry):
                return previous, False
            if previous is not None:
                self.connection.execute("UPDATE stream_explanations SET status='failed', result=? WHERE attempt_id=? AND status='generating'",
                    (_json(previous["result"]), previous["attempt_id"]))
            now, attempt = time.time(), str(uuid4())
            self.connection.execute("""INSERT INTO stream_explanations
                (attempt_id,event_id,status,created_at,lease_until,input) VALUES (?,?,'generating',?,?,?)""",
                (attempt, event_id, now, now + lease_seconds, payload))
            record = self.get_explanation(event_id)
        return record, True

    def finish_explanation(self, attempt_id, result):
        encoded = _json(result)
        state = result["status"] if result["status"] in {"ready", "insufficient_knowledge"} else "failed"
        with self._transaction():
            changed = self.connection.execute("UPDATE stream_explanations SET status=?,result=?,finished_at=? WHERE attempt_id=? AND status='generating'",
                (state, encoded, time.time(), attempt_id))
            if changed.rowcount != 1:
                raise EventConflict("explanation attempt is no longer active")
            event_id = self.connection.execute("SELECT event_id FROM stream_explanations WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
            record = self.get_explanation(event_id)
        return record
