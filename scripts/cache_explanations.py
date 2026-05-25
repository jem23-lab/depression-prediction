"""
cache_explanations.py
────────────────────────────────────────────────────────────────────
Batch-generate explanation caches for all paragraphs.
Only runs a method if the explanation is missing in training_examples.py.
"""

import argparse
import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.training_examples import (
    PARAGRAPHS,
    get_cached_explanation,
    get_cached_prediction,
    save_explanation,
    save_prediction,
)
from shared.depression_model import predict_proba, classify_severity, LABEL_MAP


logger = logging.getLogger("cache_explanations")


def _prediction_from_text(paragraph_id: str, text: str, save: bool) -> Tuple[str, float]:
    cached_label, cached_conf = get_cached_prediction(paragraph_id)
    if cached_label is not None and cached_conf is not None:
        return cached_label, cached_conf

    probs = predict_proba([text])[0]
    label, _, _ = classify_severity(probs)
    idx = int(probs.argmax()) if probs is not None else 0
    conf = float(probs[idx]) if probs is not None else 0.0
    if save:
        save_prediction(paragraph_id, label, conf)
    return label, conf


def _override_prediction(model_result, label: str) -> None:
    if hasattr(model_result, "pred_label"):
        setattr(model_result, "pred_label", label)
    if hasattr(model_result, "pred_label_idx"):
        label_to_idx = {v: k for k, v in LABEL_MAP.items()}
        setattr(model_result, "pred_label_idx", label_to_idx.get(label, 0))


def _run_shap(text: str, label: str) -> str:
    from shared.depression_model import explain_with_shap
    from architecture.shap_explainer.shap_explainer import generate_shap_explanation

    result = explain_with_shap(text)
    _override_prediction(result, label)
    return generate_shap_explanation(text, result)


def _run_rag(text: str, label: str) -> str:
    from architecture.rag_explainer.rag_explainer import run_rag_pipeline, generate_rag_explanation

    result = run_rag_pipeline(text)
    _override_prediction(result, label)
    return generate_rag_explanation(text, result)


def _run_counterfactual(text: str, label: str) -> str:
    from architecture.shap_counterfactual_explainer.cf_generator import generate_counterfactuals
    from architecture.shap_counterfactual_explainer.cf_explainer import generate_cf_explanation

    result = generate_counterfactuals(text, n_candidates=3, n_attempts=2)
    _override_prediction(result, label)
    return generate_cf_explanation(text, result)


def _run_hybrid(text: str, label: str) -> str:
    from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline import run_hybrid_pipeline
    from architecture.hybrid_shap_rag_counterfactual.hybrid_explainer import generate_hybrid_explanation

    result = run_hybrid_pipeline(text, cf_candidates=2, cf_attempts=1)
    if hasattr(result, "shap_result") and result.shap_result:
        _override_prediction(result.shap_result, label)
    if hasattr(result, "rag_result") and result.rag_result:
        _override_prediction(result.rag_result, label)
    return generate_hybrid_explanation(text, result)


_METHOD_RUNNERS: Dict[str, Callable[[str, str], str]] = {
    "SHAP": _run_shap,
    "RAG": _run_rag,
    "COUNTERFACTUAL": _run_counterfactual,
    "HYBRID": _run_hybrid,
}


def _parse_methods(methods_raw: str) -> List[str]:
    items = [m.strip().upper() for m in (methods_raw or "").split(",") if m.strip()]
    valid = [m for m in items if m in _METHOD_RUNNERS]
    return valid or list(_METHOD_RUNNERS.keys())


def _filter_paragraphs(paragraphs: Iterable[dict], only_ids: Optional[List[str]], limit: Optional[int]) -> List[dict]:
    selected = list(paragraphs)
    if only_ids:
        wanted = set(only_ids)
        selected = [p for p in selected if p.get("id") in wanted]
    if limit is not None:
        selected = selected[: max(limit, 0)]
    return selected


def run_cache(
    methods: List[str],
    only_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    save_pred: bool = False,
) -> None:
    paragraphs = _filter_paragraphs(PARAGRAPHS, only_ids, limit)
    if not paragraphs:
        logger.info("No paragraphs matched the filters.")
        return

    total_missing = 0
    total_written = 0

    for row in paragraphs:
        paragraph_id = row.get("id")
        text = row.get("text") or ""
        if not paragraph_id or not text:
            continue

        label, conf = _prediction_from_text(paragraph_id, text, save_pred)

        for method in methods:
            cached = get_cached_explanation(paragraph_id, method)
            if cached:
                logger.info("%s: %s cached", paragraph_id, method)
                continue

            total_missing += 1
            logger.info("%s: %s missing -> generating", paragraph_id, method)
            if dry_run:
                continue

            try:
                explanation = _METHOD_RUNNERS[method](text, label)
                save_explanation(paragraph_id, method, explanation)
                total_written += 1
            except Exception as exc:
                logger.exception("%s: %s failed: %s", paragraph_id, method, exc)

        if save_pred and conf is not None:
            logger.info("%s: prediction=%s (%.3f)", paragraph_id, label, conf)

    logger.info("Done. Missing=%d, Written=%d", total_missing, total_written)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache missing explanations into training_examples.py",
    )
    parser.add_argument(
        "--methods",
        default="SHAP,RAG,COUNTERFACTUAL,HYBRID",
        help="Comma-separated methods: SHAP,RAG,COUNTERFACTUAL,HYBRID",
    )
    parser.add_argument(
        "--only-ids",
        default=None,
        help="Comma-separated paragraph ids to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of paragraphs to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without calling any methods",
    )
    parser.add_argument(
        "--save-pred",
        action="store_true",
        help="Save prediction_label and prediction_confidence if missing",
    )
    args = parser.parse_args()

    only_ids = [s.strip() for s in args.only_ids.split(",")] if args.only_ids else None
    methods = _parse_methods(args.methods)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_cache(
        methods=methods,
        only_ids=only_ids,
        limit=args.limit,
        dry_run=args.dry_run,
        save_pred=args.save_pred,
    )


if __name__ == "__main__":
    main()
