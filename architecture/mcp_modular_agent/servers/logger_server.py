import os
from datetime import datetime
from fastmcp import FastMCP
import logging
from pathlib import Path
import json

# Setup basic logger for this server process
logger = logging.getLogger("LoggerServer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

LOG_DIR = Path(os.environ.get("XAI_LOG_DIR", Path.cwd() / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

server = FastMCP("LoggerServer")

ALLOWED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _session_log_path() -> Path:
    """Create a new timestamped log file for each session/request.
    The orchestrator can call this tool at session start to obtain a filename.
    """
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"session-{ts}.log"
    return LOG_DIR / fname


@server.tool()
def start_session_log() -> dict:
    """Create a new session log file and return its path (string).
    Caller should store and reuse the returned path when logging events.
    """
    try:
        p = _session_log_path()
        # ensure file exists
        p.touch(exist_ok=True)
        logger.info("Created session log: %s", p)
        return {"ok": True, "path": str(p)}
    except Exception:
        logger.exception("Failed to create session log file")
        return {"ok": False, "error": "failed to create session log"}


@server.tool()
def log_event(message: str, level: str = "INFO", payload: dict | None = None, session_log_path: str | None = None) -> dict:
    """Append a structured log line to a session-specific log file and return status.

    If session_log_path is not provided, a new timestamped file will be used.
    """
    try:
        ts = datetime.utcnow().isoformat()
        lvl = (level or "INFO").upper()
        if lvl not in ALLOWED_LEVELS:
            logger.warning("Invalid log level '%s' received; coercing to INFO", level)
            lvl = "INFO"

        if session_log_path:
            path = Path(session_log_path)
        else:
            path = _session_log_path()
            path.touch(exist_ok=True)

        record = {"ts": ts, "level": lvl, "message": message, "payload": payload or {}}
        line = json.dumps(record, ensure_ascii=False)

        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # also mirror to server stdout for convenience
        logger.info("Logged event to %s: %s", path.name, message)
        return {"ok": True, "timestamp": ts, "path": str(path)}
    except Exception:
        logger.exception("Failed to write log_event")
        return {"ok": False, "error": "failed to write log"}


if __name__ == "__main__":
    server.run()
