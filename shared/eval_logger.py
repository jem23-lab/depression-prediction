import csv
import os
from datetime import datetime, timezone


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_evaluation_row(csv_path: str, row: dict):
    _ensure_parent(csv_path)

    fieldnames = [
        "timestamp_utc",
        "user_id",
        "session_id",
        "paragraph_id",
        "paragraph_text",
        "selected_use_case",
        "selected_use_case_name",
        "prediction_label",
        "prediction_confidence",
        "explanation_text",
        "rating_clarity",
        "rating_correctness",
        "rating_helpfulness",
        "rating_overall_avg",
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        data = {k: row.get(k, "") for k in fieldnames}
        if not data.get("timestamp_utc"):
            data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        writer.writerow(data)
