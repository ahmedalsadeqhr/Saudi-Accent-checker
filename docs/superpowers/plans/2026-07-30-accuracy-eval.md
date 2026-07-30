# Accuracy Evaluation Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone script that measures the deployed dialect classifier's accuracy against labeled ground truth from the public `ArabicSpeech/ADI17` dataset, reporting overall + per-bucket accuracy and a confusion matrix, with extra weight on the Saudi/Gulf case.

**Architecture:** Two new files: `scripts/eval_metrics.py` (pure label-mapping and accuracy-aggregation functions, unit tested) and `scripts/evaluate_accuracy.py` (the runner — fetches a small stratified sample of labeled clips, classifies each with the existing `DialectClassifier`, aggregates results, prints + saves a report). No changes to `app.py` or the existing `src/` modules — they're reused as-is.

**Tech Stack:** Python (existing project stack), plus direct HTTP calls to the public `datasets-server.huggingface.co` REST API via `requests` (already an installed transitive dependency of `transformers`/`huggingface_hub`, added here as an explicit direct dependency).

## Global Constraints

- No paid/cloud APIs.
- No changes to `app.py`, `src/audio_preprocessing.py`, or `src/dialect_classifier.py` — only reused.
- Sample composition (from the design spec): `KSA` ×20, `UAE` ×10, `EGY` ×15, `JOR` ×15, `MOR` ×15 (~75 clips total), mapped to the model's buckets: `KSA`/`UAE` → `Gulf`, `EGY` → `Egyptian`, `JOR` → `Levantine`, `MOR` → `Maghrebi`.
- MSA is not evaluated against ground truth (ADI17 has no MSA samples) — this must be stated explicitly in the report output, not silently omitted.
- A clip that fails to fetch/decode/classify is logged and skipped, not fatal to the run; the report states how many clips were skipped.
- **Implementation-detail deviation from the design spec's wording:** the spec described "streaming via the `datasets` library." During planning, inspecting the dataset revealed its 5 test-split parquet shards are ~600MB each (~3GB total) and dialects appear to be stored in large contiguous blocks — so `datasets`-library streaming to collect a small stratified sample could require downloading most/all of those shards, defeating the "small download" intent. Instead, this plan uses the public `datasets-server.huggingface.co` REST API directly: it serves lightweight per-row JSON (including a fetchable audio URL) without pulling whole parquet shards, so we can page through cheaply to find matching rows and only download the ~75 actual audio files needed. Same goal (small sample, no big download), better mechanism — no scope change, no need for a new approval round.

---

### Task 1: Label mapping and accuracy aggregation

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/eval_metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Produces:
  - `scripts.eval_metrics.ADI17_LABEL_TO_BUCKET: dict[str, str]` — `{"KSA": "Gulf", "UAE": "Gulf", "EGY": "Egyptian", "JOR": "Levantine", "MOR": "Maghrebi"}`
  - `scripts.eval_metrics.SAMPLE_PLAN: dict[str, int]` — `{"KSA": 20, "UAE": 10, "EGY": 15, "JOR": 15, "MOR": 15}`
  - `scripts.eval_metrics.map_adi17_label(dialect_code: str) -> str` — raises `ValueError` for a code not in `ADI17_LABEL_TO_BUCKET`.
  - `scripts.eval_metrics.AccuracyAggregator` — class with:
    - `.record(truth_bucket: str, predicted_label: str) -> None`
    - `.record_skip() -> None`
    - `.total: int`, `.correct: int`, `.skipped: int` (plain attributes)
    - `.overall_accuracy() -> float` — `correct / total`, `0.0` if `total == 0`.
    - `.per_bucket_accuracy() -> dict[str, float]` — per truth-bucket accuracy (correct-for-that-bucket / total-for-that-bucket).
    - `.confusion_matrix_text() -> str` — a human-readable text table of truth bucket × predicted label counts.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval_metrics.py
import pytest

from scripts.eval_metrics import ADI17_LABEL_TO_BUCKET, map_adi17_label, AccuracyAggregator


def test_map_adi17_label_known_codes():
    assert map_adi17_label("KSA") == "Gulf"
    assert map_adi17_label("UAE") == "Gulf"
    assert map_adi17_label("EGY") == "Egyptian"
    assert map_adi17_label("JOR") == "Levantine"
    assert map_adi17_label("MOR") == "Maghrebi"


def test_map_adi17_label_unknown_code_raises():
    with pytest.raises(ValueError):
        map_adi17_label("XYZ")


def test_aggregator_tracks_overall_accuracy():
    aggregator = AccuracyAggregator()
    aggregator.record("Gulf", "Gulf")
    aggregator.record("Gulf", "Egyptian")
    aggregator.record("Egyptian", "Egyptian")
    assert aggregator.total == 3
    assert aggregator.correct == 2
    assert aggregator.overall_accuracy() == pytest.approx(2 / 3)


def test_aggregator_overall_accuracy_is_zero_when_empty():
    aggregator = AccuracyAggregator()
    assert aggregator.overall_accuracy() == 0.0


def test_aggregator_tracks_per_bucket_accuracy():
    aggregator = AccuracyAggregator()
    aggregator.record("Gulf", "Gulf")
    aggregator.record("Gulf", "Egyptian")
    aggregator.record("Egyptian", "Egyptian")
    per_bucket = aggregator.per_bucket_accuracy()
    assert per_bucket["Gulf"] == pytest.approx(0.5)
    assert per_bucket["Egyptian"] == pytest.approx(1.0)


def test_aggregator_confusion_matrix_text_contains_labels_and_counts():
    aggregator = AccuracyAggregator()
    aggregator.record("Gulf", "Gulf")
    aggregator.record("Gulf", "Egyptian")
    text = aggregator.confusion_matrix_text()
    assert "Gulf" in text
    assert "Egyptian" in text


def test_aggregator_record_skip_counts_separately_from_total():
    aggregator = AccuracyAggregator()
    aggregator.record_skip()
    aggregator.record_skip()
    assert aggregator.skipped == 2
    assert aggregator.total == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_metrics'` (and `scripts` itself, until Step 3 adds `scripts/__init__.py`).

- [ ] **Step 3: Implement `scripts/__init__.py` and `scripts/eval_metrics.py`**

`scripts/__init__.py` — empty file, makes `scripts` a package (mirrors `src/__init__.py`).

```python
# scripts/eval_metrics.py
ADI17_LABEL_TO_BUCKET = {
    "KSA": "Gulf",
    "UAE": "Gulf",
    "EGY": "Egyptian",
    "JOR": "Levantine",
    "MOR": "Maghrebi",
}

SAMPLE_PLAN = {
    "KSA": 20,
    "UAE": 10,
    "EGY": 15,
    "JOR": 15,
    "MOR": 15,
}


def map_adi17_label(dialect_code: str) -> str:
    try:
        return ADI17_LABEL_TO_BUCKET[dialect_code]
    except KeyError:
        raise ValueError(f"No bucket mapping for ADI17 dialect code: {dialect_code!r}")


class AccuracyAggregator:
    def __init__(self):
        self.confusion: dict[str, dict[str, int]] = {}
        self.total = 0
        self.correct = 0
        self.skipped = 0

    def record(self, truth_bucket: str, predicted_label: str) -> None:
        self.total += 1
        if predicted_label == truth_bucket:
            self.correct += 1
        bucket_row = self.confusion.setdefault(truth_bucket, {})
        bucket_row[predicted_label] = bucket_row.get(predicted_label, 0) + 1

    def record_skip(self) -> None:
        self.skipped += 1

    def overall_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def per_bucket_accuracy(self) -> dict:
        result = {}
        for truth_bucket, predictions in self.confusion.items():
            bucket_total = sum(predictions.values())
            bucket_correct = predictions.get(truth_bucket, 0)
            result[truth_bucket] = bucket_correct / bucket_total if bucket_total else 0.0
        return result

    def confusion_matrix_text(self) -> str:
        predicted_labels = sorted({label for row in self.confusion.values() for label in row})
        lines = ["Truth \\ Predicted".ljust(18) + "".join(label.ljust(12) for label in predicted_labels)]
        for truth_bucket in sorted(self.confusion):
            row = self.confusion[truth_bucket]
            lines.append(truth_bucket.ljust(18) + "".join(str(row.get(label, 0)).ljust(12) for label in predicted_labels))
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat: add label mapping and accuracy aggregation for eval script"
```

---

### Task 2: Evaluation runner script

**Files:**
- Create: `scripts/evaluate_accuracy.py`
- Modify: `requirements.txt` (add `requests>=2.32`)
- Modify: `README.md` (add an "Accuracy evaluation" section)

**Interfaces:**
- Consumes: `scripts.eval_metrics.ADI17_LABEL_TO_BUCKET`, `scripts.eval_metrics.SAMPLE_PLAN`, `scripts.eval_metrics.AccuracyAggregator`, `src.audio_preprocessing.to_mono`, `src.audio_preprocessing.resample`, `src.config.TARGET_SAMPLE_RATE`, `src.dialect_classifier.DialectClassifier`, `src.dialect_classifier.top_result`
- Produces: a runnable script (`python scripts/evaluate_accuracy.py`), no importable interface consumed elsewhere — this is the last task.

This task is not TDD'd — it's an integration script whose real behavior depends on a live network call to Hugging Face and the real ~0.3B model, which is exactly what Task 1's unit tests were designed to let us skip testing directly. Verification is manual (Step 5 below), matching how Task 4 of the original app plan handled Streamlit UI verification.

- [ ] **Step 1: Add `requests` to `requirements.txt`**

Append `requests>=2.32` to `requirements.txt` (after `pytest>=8.0`).

- [ ] **Step 2: Write `scripts/evaluate_accuracy.py`**

```python
# scripts/evaluate_accuracy.py
"""Evaluate the deployed dialect classifier's accuracy against ADI17 ground truth.

Run: python scripts/evaluate_accuracy.py
"""
import os
import sys
from datetime import date

import requests

from src.audio_preprocessing import to_mono, resample
from src.config import TARGET_SAMPLE_RATE
from src.dialect_classifier import DialectClassifier, top_result
from scripts.eval_metrics import ADI17_LABEL_TO_BUCKET, SAMPLE_PLAN, AccuracyAggregator

ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET_PARAMS = {"dataset": "ArabicSpeech/ADI17", "config": "default", "split": "test"}
PAGE_SIZE = 100
REPORT_PATH_TEMPLATE = "docs/eval/accuracy-report-{date}.md"


def find_labeled_clip_urls() -> list:
    """Pages through the datasets-server rows API to find clip URLs for SAMPLE_PLAN's labels."""
    remaining = dict(SAMPLE_PLAN)
    found = []
    offset = 0
    while any(count > 0 for count in remaining.values()):
        params = {**DATASET_PARAMS, "offset": offset, "length": PAGE_SIZE}
        response = requests.get(ROWS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
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
    aggregator = evaluate()
    report = build_report(aggregator)
    print("\n" + report)

    report_path = REPORT_PATH_TEMPLATE.format(date=date.today().isoformat())
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add a "datasets-server API unreachable" guard**

Wrap the initial `find_labeled_clip_urls()` call in `evaluate()` so that if the very first request to `datasets-server.huggingface.co` fails (e.g. no internet), the script exits with a clear message rather than a raw traceback. Edit `evaluate()`'s call site in `main()`:

```python
def main() -> None:
    try:
        aggregator = evaluate()
    except requests.exceptions.RequestException as exc:
        print(f"Could not reach the Hugging Face datasets-server API: {exc}", file=sys.stderr)
        sys.exit(1)

    report = build_report(aggregator)
    ...
```

(Only the first line of `evaluate()`'s per-page requests can raise this uncaught — per-clip audio-download failures are already caught inside the loop and recorded as skips.)

- [ ] **Step 4: Add the README section**

Append to `README.md`:

```markdown

## Accuracy evaluation

To measure the deployed model's accuracy against labeled ground truth from the public
`ArabicSpeech/ADI17` dataset (weighted toward the Saudi/Gulf case):

\`\`\`bash
pip install requests  # if not already installed
python scripts/evaluate_accuracy.py
\`\`\`

This fetches ~75 labeled clips over the network (no local dataset download), classifies
each with the current model, and writes a report to `docs/eval/accuracy-report-<date>.md`
with overall accuracy, per-dialect accuracy, and a confusion matrix. Note: Modern Standard
Arabic (MSA) isn't evaluated against ground truth — the source dataset has no MSA samples.
```

- [ ] **Step 5: Manually verify the script runs end-to-end**

Run: `python scripts/evaluate_accuracy.py`
Expected: prints "Found 75 labeled clips to evaluate." (or close to 75, allowing for any
skipped/undecodable clips), then an accuracy report with overall accuracy, per-bucket
accuracy, and a confusion matrix, and confirms the report was saved under `docs/eval/`.
This step needs internet access and will take a few minutes (network calls + ~75 CPU
inferences).

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_accuracy.py requirements.txt README.md
git commit -m "feat: add accuracy evaluation script against ADI17 ground truth"
```

---

## Self-Review Notes

- **Spec coverage:** ADI17 data source with streaming-friendly fetch (Task 2, via the lighter REST-API mechanism — see the Global Constraints deviation note), the exact sample composition from the spec (Task 2's `SAMPLE_PLAN`), label mapping to model buckets (Task 1), overall + per-bucket accuracy + confusion matrix (Task 1 aggregation, Task 2 report), MSA gap called out explicitly in the report (Task 2), error handling that skips-and-counts rather than crashing (Task 2), unit tests for the pure logic without needing the real model/dataset (Task 1) — all covered. No changes to `app.py`/`src/` — confirmed, Task 2 only adds/modifies `scripts/`, `requirements.txt`, `README.md`.
- **Type consistency:** `map_adi17_label`/`ADI17_LABEL_TO_BUCKET` produce the same bucket strings (`"Gulf"`, `"Egyptian"`, `"Levantine"`, `"Maghrebi"`) that `AccuracyAggregator.record`'s `truth_bucket` parameter expects and that `DialectClassifier.predict`/`top_result` (from the existing codebase) already produce as `predicted_label` for comparison — both sides speak the model's real label vocabulary (`Maghrebi`, `MSA`, `Egyptian`, `Gulf`, `Levantine`). `AccuracyAggregator`'s attributes/methods used in `scripts/evaluate_accuracy.py` (`.record`, `.record_skip`, `.correct`, `.total`, `.skipped`, `.overall_accuracy()`, `.per_bucket_accuracy()`, `.confusion_matrix_text()`) all match what Task 1 defines.
- **Out of scope confirmed absent:** no paid APIs, no CI integration, no MSA ground-truth fabrication, no modification of the existing app/src modules anywhere in this plan.
