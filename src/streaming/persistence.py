"""
告警与段判定的 SQLite 持久化
=============================

**为什么必须有**：原实现 ``engine.py`` 用 ``deque(maxlen=200)`` 只把告警放在
内存里，**进程重启即全丢**。运维人员会问「昨天有哪些异常」「上周有没有
异常峰值」，答不上来。演示可以，交付不行。

用标准库 ``sqlite3``，不引入新依赖。库文件默认 ``data/alerts.db``。

## 设计

**双写**：内存 deque 继续维护（WebSocket 广播从内存读，行为不变），
**额外**落 SQLite。落库失败**只记日志不抛异常** —— 数据库故障不应让
实时检测链路整体挂掉（检测是主路径，告警持久化是副路径）。

**幂等建表**：``CREATE TABLE IF NOT EXISTS``，启动时调用，不 DROP。
段表以 ``(segment, channel)`` 为主键，同一段重复判定时
``INSERT OR REPLACE``，避免重复累积。

**检索接口**：``/api/alerts`` 支持按时间范围/通道/严重度过滤 + 分页。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "alerts.db"

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: Optional[Path] = None


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")  # 并发读不阻塞写
    return conn


def init_db(path: Path | str = DEFAULT_DB) -> Path:
    """幂等建表。返回库路径。

    ``check_same_thread=False`` + 模块级锁：FastAPI 是多线程，
    sqlite3 连接默认不能跨线程用，这里配合 ``_lock`` 保证串行访问。
    """
    global _conn, _db_path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if _conn is not None and _db_path == p:
            return p
        _conn = _connect(p)
        _db_path = p
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL    NOT NULL,
                segment      TEXT    NOT NULL,
                channel      TEXT    NOT NULL,
                channel_label TEXT,
                severity     TEXT,
                anomaly_score REAL,
                rule_hits    INTEGER,
                first_ts     REAL,
                last_ts      REAL,
                official     INTEGER DEFAULT 0,
                payload      TEXT,
                created_at   REAL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_ts      ON alerts(ts);
            CREATE INDEX IF NOT EXISTS idx_alerts_channel ON alerts(channel);
            CREATE INDEX IF NOT EXISTS idx_alerts_sev     ON alerts(severity);

            CREATE TABLE IF NOT EXISTS segment_verdicts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                segment    TEXT    NOT NULL,
                channel    TEXT    NOT NULL,
                ts         REAL    NOT NULL,
                n_points   INTEGER,
                is_anomaly INTEGER,
                severity   TEXT,
                score      REAL,
                features   TEXT,
                created_at REAL,
                UNIQUE(segment, channel)
            );
            CREATE INDEX IF NOT EXISTS idx_verdict_ts ON segment_verdicts(ts);
            """
        )
        _conn.commit()
    return p


def _ready() -> bool:
    return _conn is not None


def save_alert(alert: dict[str, Any]) -> bool:
    """落库一条告警。失败只记日志，不抛异常。

    返回是否成功。
    """
    if not _ready():
        return False
    try:
        with _lock:
            assert _conn is not None
            _conn.execute(
                """INSERT INTO alerts
                   (ts, segment, channel, channel_label, severity, anomaly_score,
                    rule_hits, first_ts, last_ts, official, payload, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    float(alert.get("last_ts", time.time())),
                    str(alert.get("segment", "")),
                    str(alert.get("channel", "")),
                    alert.get("channel_label"),
                    alert.get("severity"),
                    alert.get("anomaly_score"),
                    alert.get("hits"),
                    alert.get("first_ts"),
                    alert.get("last_ts"),
                    1 if str(alert.get("segment", "")).startswith("LIVE-") is False else 0,
                    json.dumps(alert, ensure_ascii=False),
                    time.time(),
                ),
            )
            _conn.commit()
        return True
    except Exception as e:  # noqa: BLE001 — 持久化是副路径，不应影响主链路
        print(f"[alerts_db] 告警落库失败（不影响检测）: {type(e).__name__}: {e}")
        return False


def save_verdict(
    seg_id: str,
    channel: str,
    ts: float,
    n_points: int,
    is_anomaly: bool,
    severity: str,
    score: float,
    features: dict[str, Any],
) -> bool:
    """落库一条段判定结果（无论是否异常，便于事后回溯）。"""
    if not _ready():
        return False
    try:
        with _lock:
            assert _conn is not None
            _conn.execute(
                """INSERT OR REPLACE INTO segment_verdicts
                   (segment, channel, ts, n_points, is_anomaly, severity,
                    score, features, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    str(seg_id),
                    str(channel),
                    float(ts),
                    int(n_points),
                    1 if is_anomaly else 0,
                    severity,
                    float(score),
                    json.dumps(features, ensure_ascii=False),
                    time.time(),
                ),
            )
            _conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[alerts_db] 段判定落库失败（不影响检测）: {type(e).__name__}: {e}")
        return False


def query_alerts(
    since: Optional[float] = None,
    until: Optional[float] = None,
    channel: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """按条件检索告警。返回 (当前页, 符合条件的总数)。"""
    if not _ready():
        return [], 0
    where: list[str] = []
    params: list[Any] = []
    if since is not None:
        where.append("ts >= ?")
        params.append(float(since))
    if until is not None:
        where.append("ts <= ?")
        params.append(float(until))
    if channel:
        where.append("channel = ?")
        params.append(channel)
    if severity:
        where.append("severity = ?")
        params.append(severity)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    with _lock:
        assert _conn is not None
        total = _conn.execute(
            f"SELECT COUNT(*) FROM alerts {clause}", params
        ).fetchone()[0]
        rows = _conn.execute(
            f"""SELECT id, ts, segment, channel, channel_label, severity,
                       anomaly_score, rule_hits, first_ts, last_ts, payload
                FROM alerts {clause}
                ORDER BY ts DESC, id DESC
                LIMIT ? OFFSET ?""",
            [*params, int(limit), int(offset)],
        ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r[10]) if r[10] else {}
        except json.JSONDecodeError:
            payload = {}
        out.append(
            {
                "id": r[0],
                "ts": r[1],
                "segment": r[2],
                "channel": r[3],
                "channel_label": r[4],
                "severity": r[5],
                "anomaly_score": r[6],
                "hits": r[7],
                "first_ts": r[8],
                "last_ts": r[9],
                "summary": payload.get("summary", ""),
                "payload": payload,
            }
        )
    return out, int(total)


def stats() -> dict[str, Any]:
    """给 /api/alerts/stats 用的汇总。"""
    if not _ready():
        return {"alerts": 0, "verdicts": 0, "by_severity": {}, "by_channel": {}}
    with _lock:
        assert _conn is not None
        a = _conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        v = _conn.execute("SELECT COUNT(*) FROM segment_verdicts").fetchone()[0]
        sev = dict(
            _conn.execute(
                "SELECT severity, COUNT(*) FROM alerts GROUP BY severity"
            ).fetchall()
        )
        ch = dict(
            _conn.execute(
                "SELECT channel, COUNT(*) FROM alerts GROUP BY channel"
            ).fetchall()
        )
    return {"alerts": int(a), "verdicts": int(v), "by_severity": sev, "by_channel": ch}
