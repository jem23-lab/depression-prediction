import os
from datetime import datetime
from fastmcp import FastMCP

LOG_DIR = os.environ.get("XAI_LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

server = FastMCP("logger-servers")

@server.tool()
def log_event(message: str, level: str = "INFO", payload: dict | None = None) -> dict:
    ts = datetime.utcnow().isoformat()
    line = f"{ts} {level} {message} {payload or {}}\n"
    with open(os.path.join(LOG_DIR, "session.log"), "a", encoding="utf-8") as f:
        f.write(line)
    return {"ok": True, "timestamp": ts}

if __name__ == "__main__":
    server.run()
