"""Verify the real local HTTP/WS server against an isolated SQLite database."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def verify(database):
    if database.exists():
        raise ValueError("use a new isolated database path")
    database.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = str(uuid4())
    env = {**os.environ, "API_AUTH_KEYS": key, "API_AUTH_DISABLED": "false",
           "API_RATE_LIMIT_DISABLED": "false", "MICROSAT_EVENTS_DB": str(database), "PYTHONIOENCODING": "utf-8"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"
    def request(path, body=None, *, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["X-API-Key"] = key
        req = urllib.request.Request(base + path, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        try:
            with opener.open(req, timeout=15) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())
    with (database.parent / "wire-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning", "--no-access-log"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("local verification server exited")
                try:
                    if request("/api/health", authenticated=False)[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError("local verification server did not start")
            assert request("/api/stream/state", authenticated=False)[0] == 401
            uri = f"ws://127.0.0.1:{port}/api/stream/ws?api_key={key}"
            with connect(uri, proxy=None, open_timeout=15) as ws:
                assert json.loads(ws.recv(timeout=15))["type"] == "snapshot"
                code, session = request("/api/stream/live/start", {"source_id": "wire-verification"})
                assert code == 200
                sid = session["session_id"]
                batch = {"session_id": sid, "batch_id": str(uuid4()), "frames": [
                    {"ts": float(i * 5), "channel": "CADC0884", "value": value, "sampling": 5, "segment_id": 7}
                    for i, value in enumerate([1e9, -1e9, 1e9])]}
                code, accepted = request("/api/stream/ingest", batch)
                assert code == 200 and accepted["accepted"] == 3
                assert request("/api/stream/ingest", batch)[1] == accepted
                message = json.loads(ws.recv(timeout=15))
                assert message["type"] == "batch" and message["stats"]["total_points"] == 3
                ended = request("/api/stream/live/end", {"session_id": sid})[1]
                assert request("/api/stream/live/end", {"session_id": sid})[1] == ended
                event = ended["closed_events"][0]
                assert event["status"] == "judged" and len(event["features"]) == 18
                assert event["judgement"]["final_pred"] is True
                assert request(f"/api/stream/events/{event['event_id']}")[1] == event
                assert request("/api/stream/alerts/stats")[1]["alerts"] == 1
            with connect(uri, proxy=None, open_timeout=15) as ws:
                state = json.loads(ws.recv(timeout=15))["state"]
                assert state["stats"]["judged_segments"] == state["stats"]["alerts"] == 1
                assert state["session_id"] == sid
            return {"status": "passed", "transport": "local TCP HTTP and WebSocket",
                    "auth_enabled": True, "missing_http_key_status": 401,
                    "snapshot_batch_reconnect": True, "batch_retry": True, "end_retry": True,
                    "event_detail_matches": True, "points": 3, "judgements": 1, "alerts": 1,
                    "event_id": event["event_id"], "session_id": sid,
                    "bundle_id": event["judgement"]["bundle_id"], "database": str(database)}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
