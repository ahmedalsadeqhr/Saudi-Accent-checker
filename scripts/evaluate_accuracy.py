# scripts/evaluate_accuracy.py
"""Evaluate the deployed dialect classifier's accuracy against ADI17 ground truth.

Run: python scripts/evaluate_accuracy.py
"""
import io
import os
import sys
import time
from datetime import date, datetime

import requests

# Allow running this script directly (`python scripts/evaluate_accuracy.py`) regardless
# of the caller's working directory: `python <path>` only puts the script's own directory
# (scripts/) on sys.path, not the repo root, so `src`/`scripts` package imports below would
# otherwise fail with ModuleNotFoundError. Running via `python -m scripts.evaluate_accuracy`
# doesn't need this, but the plain script form does.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.audio_preprocessing import prepare_audio
from src.config import DEFAULT_MODEL_SOURCE
from src.dialect_classifier import DialectClassifier, top_result
from scripts.eval_metrics import SAMPLE_PLAN, AccuracyAggregator, map_adi17_label

ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET_PARAMS = {"dataset": "ArabicSpeech/ADI17", "config": "default", "split": "test"}
PAGE_SIZE = 100
REPORT_PATH_TEMPLATE = os.path.join(REPO_ROOT, "docs", "eval", "accuracy-report-{date}.md")
MAX_PAGE_RETRIES = 8
MAX_BACKOFF_SEC = 30
PAGE_REQUEST_DELAY_SEC = 0.5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _get_with_retry(url: str, params: dict | None = None) -> requests.Response:
    """GETs a URL, retrying with backoff on transient network/server failures.

    Observed in practice while building this script: the public datasets-server API
    enforces an undocumented burst rate limit (429, no Retry-After header) that a
    straight paging loop can trip when fetching many pages back-to-back, occasionally
    stalls past a 30s read timeout under load, and intermittently returns 502 Bad
    Gateway partway through a long paging run. All are transient and worth a retry
    rather than aborting the whole run. The same retry behavior applies to per-clip
    audio downloads, which hit the same kind of transient failures.
    """
    backoff_sec = 2
    last_error: Exception | None = None
    for attempt in range(MAX_PAGE_RETRIES):
        is_last_attempt = attempt == MAX_PAGE_RETRIES - 1
        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if is_last_attempt:
                raise
            print(
                f"Request to {url} failed ({exc}) "
                f"(attempt {attempt + 1}/{MAX_PAGE_RETRIES}); retrying in {backoff_sec}s.",
                file=sys.stderr,
            )
            time.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2, MAX_BACKOFF_SEC)
            continue
        if response.status_code in RETRYABLE_STATUS_CODES and not is_last_attempt:
            print(
                f"Request to {url} returned HTTP {response.status_code} "
                f"(attempt {attempt + 1}/{MAX_PAGE_RETRIES}); retrying in {backoff_sec}s.",
                file=sys.stderr,
            )
            time.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2, MAX_BACKOFF_SEC)
            continue
        response.raise_for_status()
        return response
    raise last_error or requests.exceptions.RequestException(
        f"Exhausted retries against {url}"
    )


def _get_page_with_retry(params: dict) -> dict:
    """GETs one page of rows from the datasets-server rows API, retrying on transient failures."""
    return _get_with_retry(ROWS_API_URL, params=params).json()


def _download_audio_with_retry(audio_url: str) -> bytes:
    """Downloads a clip's audio bytes, retrying on transient failures."""
    return _get_with_retry(audio_url).content


def find_labeled_clip_urls() -> list:
    """Pages through the datasets-server rows API to find clip URLs for SAMPLE_PLAN's labels."""
    remaining = dict(SAMPLE_PLAN)
    found = []
    offset = 0
    while any(count > 0 for count in remaining.values()):
        params = {**DATASET_PARAMS, "offset": offset, "length": PAGE_SIZE}
        payload = _get_page_with_retry(params)
        rows = payload.get("rows", [])
        if not rows:
            break
        for entry in rows:
            dialect_code = entry["row"]["dialect"]
            if remaining.get(dialect_code, 0) > 0:
                audio_url = entry["row"]["audio"][0]["src"]
                found.append((dialect_code, audio_url))
                remaining[dialect_code] -= 1
        offset += PAGE_SIZE
        if offset >= payload.get("num_rows_total", float("inf")):
            break
        time.sleep(PAGE_REQUEST_DELAY_SEC)
    return found


def evaluate(clip_urls: list) -> AccuracyAggregator:
    classifier = DialectClassifier()
    aggregator = AccuracyAggregator()

    for dialect_code, audio_url in clip_urls:
        try:
            truth_bucket = map_adi17_label(dialect_code)
            audio_bytes = _download_audio_with_retry(audio_url)
            resampled = prepare_audio(io.BytesIO(audio_bytes))

            scores = classifier.predict(resampled)
            predicted_label, _ = top_result(scores)
            aggregator.record(truth_bucket, predicted_label)
        except Exception as exc:
            print(f"Skipping a {dialect_code} clip due to error: {exc}", file=sys.stderr)
            aggregator.record_skip()

    return aggregator


def build_report(aggregator: AccuracyAggregator, clip_urls_found: int) -> str:
    lines = ["# Accuracy Evaluation Report", ""]
    lines.append(f"Model: {DEFAULT_MODEL_SOURCE}")
    lines.append("Dataset: ArabicSpeech/ADI17 (test split, via datasets-server.huggingface.co)")
    lines.append(f"Sample plan: {SAMPLE_PLAN}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    sample_plan_total = sum(SAMPLE_PLAN.values())
    if clip_urls_found < sample_plan_total:
        lines.append(
            f"Note: only found {clip_urls_found}/{sample_plan_total} planned clips "
            "(dataset exhausted or match not found for some labels)."
        )
        lines.append("")
    lines.append(f"Overall accuracy: {aggregator.overall_accuracy():.1%} ({aggregator.correct}/{aggregator.total})")
    lines.append(f"Skipped clips: {aggregator.skipped}")
    lines.append("")
    lines.append("## Per-bucket accuracy")
    for bucket, accuracy in sorted(aggregator.per_bucket_accuracy().items()):
        lines.append(f"- {bucket}: {accuracy:.1%}")
    lines.append("")
    lines.append("## Confusion matrix (truth bucket rows, predicted label columns)")
    lines.append("```")
    lines.append(aggregator.confusion_matrix_text())
    lines.append("```")
    lines.append("")
    lines.append(
        "Note: MSA is not evaluated against ground truth here (the ADI17 source dataset "
        "has no MSA samples); it may still appear as a (mis)prediction in the confusion matrix."
    )
    return "\n".join(lines)


def main() -> None:
    try:
        clip_urls = find_labeled_clip_urls()
    except requests.exceptions.RequestException as exc:
        print(f"Could not reach the Hugging Face datasets-server API: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(clip_urls)} labeled clips to evaluate.")

    aggregator = evaluate(clip_urls)

    if aggregator.total == 0:
        print(
            "Every clip failed to evaluate (0/0 scored) — refusing to write a misleading "
            "0.0% report. Check network connectivity, the datasets-server API, and the "
            "classifier's model source.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = build_report(aggregator, clip_urls_found=len(clip_urls))
    print("\n" + report)

    report_path = REPORT_PATH_TEMPLATE.format(date=date.today().isoformat())
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
