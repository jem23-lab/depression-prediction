import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Iterable


SCHEMA_VERSION = "1.0"
SYSTEM_AGENTIC_XAI = "agentic_xai"
SYSTEM_MENTALLAMA = "mentallama"
SYSTEMS = (SYSTEM_AGENTIC_XAI, SYSTEM_MENTALLAMA)

QUESTION_TEXTS = {
    "shap": "Which text parts mattered most?",
    "counterfactual": "What could change the prediction?",
    "rag": "Which symptoms mattered?",
    "hybrid": "What evidence supports it?",
}

AGENTIC_TOOL_BY_QUESTION = {
    "shap": "shap",
    "counterfactual": "counterfactual",
    "rag": "rag",
    "hybrid": "hybrid",
}


class StudyResponseNotFoundError(KeyError):
    """Raised when a requested saved study response is unavailable."""


def stable_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def empty_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question_catalog": deepcopy(QUESTION_TEXTS),
        "records": [],
    }


def load_payload(path: str) -> dict:
    if not os.path.exists(path):
        return empty_payload()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    if "question_catalog" not in payload:
        payload["question_catalog"] = payload.pop("questions", deepcopy(QUESTION_TEXTS))
    payload.setdefault("records", [])
    return payload


class StudyResponseStore:
    def __init__(self, path: str):
        self.path = path
        self.payload = load_payload(path)
        self._records_by_passage: dict[str, dict] = {}
        self._index: dict[tuple[str, str, str], dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        self._records_by_passage.clear()
        self._index.clear()
        question_type_by_text = {
            text: question_type
            for question_type, text in (self.payload.get("question_catalog") or QUESTION_TEXTS).items()
        }
        for record in self.payload.get("records", []):
            passage_id = record.get("id")
            if not passage_id:
                continue
            self._records_by_passage[passage_id] = record
            for question in record.get("responses", record.get("questions", [])):
                question_type = question.get("type") or question_type_by_text.get(question.get("question"))
                if not question_type:
                    continue
                for system in SYSTEMS:
                    entry = question.get(system) or {}
                    if entry.get("status") == "success" and entry.get("response"):
                        self._index[(passage_id, question_type, system)] = {
                            "record": record,
                            "question": question,
                            "system": system,
                            "entry": entry,
                        }

    def reload(self) -> None:
        self.payload = load_payload(self.path)
        self._build_index()

    def get_response(self, passage_id: str, question_type: str, system: str) -> str:
        return self.get_record(passage_id, question_type, system)["entry"]["response"]

    def has_response(self, passage_id: str, question_type: str, system: str) -> bool:
        return (passage_id, question_type, system) in self._index

    def get_record(self, passage_id: str, question_type: str, system: str) -> dict:
        key = (passage_id, question_type, system)
        if key not in self._index:
            raise StudyResponseNotFoundError(
                f"Missing saved response for passage_id={passage_id!r}, "
                f"question_type={question_type!r}, system={system!r}."
            )
        return self._index[key]

    def get_passage_record(self, passage_id: str) -> dict:
        if passage_id not in self._records_by_passage:
            raise StudyResponseNotFoundError(f"Missing passage record for passage_id={passage_id!r}.")
        return self._records_by_passage[passage_id]

    def list_missing_records(
        self,
        passage_ids: Iterable[str] | None = None,
        question_types: Iterable[str] | None = None,
        systems: Iterable[str] | None = None,
    ) -> list[dict]:
        passage_ids = list(passage_ids) if passage_ids is not None else sorted(self._records_by_passage)
        question_types = list(question_types) if question_types is not None else list(QUESTION_TEXTS)
        systems = list(systems) if systems is not None else list(SYSTEMS)

        missing = []
        for passage_id in passage_ids:
            for question_type in question_types:
                for system in systems:
                    if not self.has_response(passage_id, question_type, system):
                        missing.append({
                            "passage_id": passage_id,
                            "question_type": question_type,
                            "system": system,
                        })
        return missing

    def validate_complete(
        self,
        passage_ids: Iterable[str],
        question_types: Iterable[str] | None = None,
        systems: Iterable[str] | None = None,
    ) -> dict:
        passage_ids = list(passage_ids)
        question_types = list(question_types) if question_types is not None else list(QUESTION_TEXTS)
        systems = list(systems) if systems is not None else list(SYSTEMS)
        missing = self.list_missing_records(passage_ids, question_types, systems)
        expected = len(passage_ids) * len(question_types) * len(systems)
        return {
            "complete": not missing,
            "expected": expected,
            "available": expected - len(missing),
            "missing": missing,
        }
