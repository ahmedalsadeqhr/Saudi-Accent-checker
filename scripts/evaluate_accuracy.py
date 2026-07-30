# scripts/evaluate_accuracy.py
"""Evaluate the deployed dialect classifier's accuracy against ADI17 ground truth.

Run: python scripts/evaluate_accuracy.py
"""
import os
import sys
import time
from datetime import date

import requests

# Allow running this script directly (`python scripts/evaluate_accuracy.py`) regardless
# of the caller's working directory: `python <path>` only puts the script's own directory
# (scripts/) on sys.path, not the repo root, so `src`/`scripts` package imports below would
# otherwise fail with ModuleNotFoundError. Running via `python -m scripts.evaluate_accuracy`
# doesn't need this, but the plain script form does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_preprocessing import to_mono, resample
from src.config import TARGET_SAMPLE_RATE
from src.dialect_classifier import DialectClassifier, top_result
from scripts.eval_metrics import ADI17_LABEL_TO_BUCKET, SAMPLE_PLAN, AccuracyAggregator

ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET_PARAMS = {"dataset": "ArabicSpeech/ADI17", "config": "default", "split": "test"}
PAGE_SIZE = 100
REPORT_PATH_TEMPLATE = "docs/eval/accuracy-report-{date}.md"
MAX_PAGE_RETRIES = 8
MAX_BACKOFF_SEC = 30
PAGE_REQUEST_DELAY_SEC = 0.5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _get_page_with_retry(params: dict) -> dict:
    """GETs one page of rows, retrying with backoff on transient datasets-server failures.

    Observed in practice while building this script: the public datasets-server API
    enforces an undocumented burst rate limit (429, no Retry-After header) that a
    straight paging loop can trip when fetching many pages back-to-back, occasionally
    stalls past a 30s read timeout under load, and intermittently returns 502 Bad
    Gateway partway through a long paging run. All are transient and worth a retry
    rather than aborting the whole run.
    """
    backoff_sec = 2
    last_error: Exception | None = None
    for attempt in range(MAX_PAGE_RETRIES):
        is_last_attempt = attempt == MAX_PAGE_RETRIES - 1
        try:
            response = requests.get(ROWS_API_URL, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if is_last_attempt:
                raise
            print(
                f"datasets-server request failed ({exc}) "
                f"(attempt {attempt + 1}/{MAX_PAGE_RETRIES}); retrying in {backoff_sec}s.",
                file=sys.stderr,
            )
            time.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2, MAX_BACKOFF_SEC)
            continue
        if response.status_code in RETRYABLE_STATUS_CODES and not is_last_attempt:
            print(
                f"datasets-server returned HTTP {response.status_code} "
                f"(attempt {attempt + 1}/{MAX_PAGE_RETRIES}); retrying in {backoff_sec}s.",
                file=sys.stderr,
            )
            time.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2, MAX_BACKOFF_SEC)
            continue
        response.raise_for_status()
        return response.json()
    raise last_error or requests.exceptions.RequestException(
        "Exhausted retries against datasets-server rows API"
    )


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
        if offset >= payload.get("num_rows_total", offset):
            break
        time.sleep(PAGE_REQUEST_DELAY_SEC)
    return found


def evaluate() -> AccuracyAggregator:
    classifier = DialectClassifier()
    aggregator = AccuracyAggregator()

    clip_urls = find_labeled_clip_urls()
    print(f"Found {len(clip_urls)} labeled clips to evaluate.")

    for dialect_code, audio_url in clip_urls:
        try:
            truth_bucket = ADI17_LABEL_TO_BUCKET[dialect_code]
            audio_response = requests.get(audio_url, timeout=30)
            audio_response.raise_for_status()

            import io
            import soundfile as sf

            waveform, sample_rate = sf.read(io.BytesIO(audio_response.content), dtype="float32")
            mono = to_mono(waveform)
            resampled = resample(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

            scores = classifier.predict(resampled)
            predicted_label, _ = top_result(scores)
            aggregator.record(truth_bucket, predicted_label)
        except Exception as exc:
            print(f"Skipping a {dialect_code} clip due to error: {exc}", file=sys.stderr)
            aggregator.record_skip()

    return aggregator


def build_report(aggregator: AccuracyAggregator) -> str:
    lines = ["# Accuracy Evaluation Report", ""]
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
        aggregator = evaluate()
    except requests.exceptions.RequestException as exc:
        print(f"Could not reach the Hugging Face datasets-server API: {exc}", file=sys.stderr)
        sys.exit(1)

    report = build_report(aggregator)
    print("\n" + report)

    report_path = REPORT_PATH_TEMPLATE.format(date=date.today().isoformat())
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
