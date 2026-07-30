# Accuracy Evaluation Script — Design

## Purpose

The Saudi Accent Checker was validated manually with real speech and looked reasonable,
but there's no objective accuracy number. This adds a standalone evaluation script that
measures the deployed model's accuracy against labeled ground truth, with particular
attention to the Saudi/Gulf case since that's the tool's headline feature.

## Data Source

`ArabicSpeech/ADI17` on Hugging Face — a public, non-gated dataset of Arabic dialect
audio labeled by country (17 countries), available in Parquet format that supports
streaming. We stream a small stratified sample from its ~12.2k-row test split — no full
263GB download.

**Known gap:** ADI17 is dialectal-only, so it has no Modern Standard Arabic (MSA) samples.
The model's 5th label (`MSA`) is therefore not evaluated against ground truth in this
pass — only observed as a possible (mis)prediction on dialectal clips. This is called out
explicitly in the eval report, not silently glossed over.

## Sample Composition

~75 clips total, weighted toward the Saudi/Gulf case since that's the core feature:

| ADI17 country label | Count | Maps to model bucket |
|---|---|---|
| `KSA` | 20 | Gulf |
| `UAE` | 10 | Gulf |
| `EGY` | 15 | Egyptian |
| `JOR` | 15 | Levantine |
| `MOR` | 15 | Maghrebi |

## Architecture

A standalone script, `scripts/evaluate_accuracy.py`, separate from the app:

1. Stream ADI17's test split with `datasets.load_dataset(..., streaming=True)`, filtering
   per label and taking the configured count for each (`itertools.islice` per label).
2. For each clip: take the raw audio array + sample rate from the dataset, convert to
   mono/16kHz using the same functions `src/audio_preprocessing.py` already exposes
   (`to_mono`, `resample`) — no new preprocessing logic.
3. Run `DialectClassifier.predict()` (reused as-is from `src/dialect_classifier.py`),
   take `top_result()`.
4. Map the clip's ADI17 country label to its ground-truth model bucket via a small
   `ADI17_LABEL_TO_BUCKET` lookup table (pure function, unit-testable).
5. Compare predicted label vs. ground-truth bucket; accumulate into a confusion matrix
   (truth bucket × predicted label) and per-bucket / overall accuracy counts (pure
   aggregation functions, unit-testable).
6. Print a console summary (overall accuracy, per-bucket accuracy, confusion matrix as a
   text table) and write the same content to `docs/eval/accuracy-report-<date>.md`, so
   results are kept for comparison across future model or fine-tuning changes.

## Error Handling

- A clip that fails to stream, decode, or classify is logged as a warning and skipped —
  not fatal to the run. The final report states how many clips were skipped and why.
- If the `datasets` library isn't installed, fail immediately with a clear message
  instructing `pip install datasets`, rather than a deep stack trace mid-run.
- Network/streaming errors from Hugging Face are caught the same way as decode errors —
  logged, skipped, counted.

## Testing

- `ADI17_LABEL_TO_BUCKET` mapping and the accuracy/confusion-matrix aggregation logic
  are pure functions, unit tested with fake prediction data (no real model or dataset
  download needed) — fast, included in the normal `pytest -m "not integration"` run.
- The end-to-end streaming + classification run is exercised manually — it's inherently
  slow and network-dependent, not part of the automated test suite. It's a script you
  run when you want a fresh accuracy number, not on every commit.

## Explicitly Out of Scope

- No MSA ground-truth evaluation (no source dataset for it — noted as a known gap).
- No CI integration — this is a manually-run script, not a gate on every change.
- No changes to `app.py`, `src/audio_preprocessing.py`, or `src/dialect_classifier.py` —
  the eval script only reuses them.
- No new paid APIs or services.
