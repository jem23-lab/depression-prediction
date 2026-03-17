import os
from datetime import datetime
from fastmcp import FastMCP
import logging
from pathlib import Path

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


@server.tool()
def log_event(message: str, level: str = "INFO", payload: dict | None = None) -> dict:
    """Append a structured log line to the shared log file and return status.

    Returns: {'ok': True, 'timestamp': iso8601} on success, or {'ok': False, 'error': str}
    """
    try:
        ts = datetime.utcnow().isoformat()
        lvl = (level or "INFO").upper()
        if lvl not in ALLOWED_LEVELS:
            logger.warning("Invalid log level '%s' received; coercing to INFO", level)
            lvl = "INFO"
        line = f"{ts} {lvl} {message} {payload or {}}\n"
        with open(LOG_DIR / "session.log", "a", encoding="utf-8") as f:
            f.write(line)
        return {"ok": True, "timestamp": ts}
    except Exception:
        logger.exception("Failed to write log_event")
        return {"ok": False, "error": "failed to write log"}


if __name__ == "__main__":
    server.run()
