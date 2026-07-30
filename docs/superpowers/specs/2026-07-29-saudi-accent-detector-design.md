# Saudi Accent Detector — Design

## Purpose

A tool to listen to someone's spoken Arabic (live mic or uploaded recording) and determine
whether their accent is Saudi/Gulf Arabic, with confidence scores across other Arabic
dialects for context. Used internally by a small team — no accounts, no public hosting,
no paid APIs.

## Architecture

Single Streamlit app (`app.py`), no separate frontend/backend split:

- **UI**: `st.audio_input` for live mic recording, `st.file_uploader` for uploaded audio
  files (wav/mp3/m4a).
- **Pipeline**: uploaded/recorded audio → preprocessing (resample to 16kHz mono via
  `pydub`/`ffmpeg`) → pretrained Arabic dialect-ID model → per-dialect confidence scores →
  displayed as a table/bar chart, with the Saudi/Gulf class highlighted.
- **Hosting**: run locally via `streamlit run app.py`, accessed by the team over the local
  network (machine IP:port). No deployment infra for v1.

## Model & Fine-Tuning Path

- **Initial model**: a pretrained open-source Arabic dialect identification model from
  Hugging Face, trained on a multi-dialect corpus (e.g. ADI17, which includes Gulf/Saudi
  Arabic among ~17 dialects). Exact model selection happens during implementation — check
  what's currently available/maintained on the Hub at build time.
- **Output**: probability per dialect class the chosen model was trained on. The
  Saudi/Gulf class is treated as the primary answer; other classes are shown as context,
  not filtered out.
- **Explanation**: v1 ships with label + confidence scores only. No generated natural-
  language explanation of *why* (e.g. specific phonetic markers) — that's an explicit
  future enhancement, not in scope now.
- **Fine-tuning path (future)**: since accuracy will improve by fine-tuning on our own
  labeled data over time:
  - A local folder holds our own labeled recordings (Saudi vs. not-Saudi).
  - A separate offline fine-tuning script (outside the Streamlit app) fine-tunes the
    pretrained model's classification head on this data and saves a new checkpoint.
  - The Streamlit app loads whichever checkpoint is at a configured `MODEL_PATH`, so
    swapping in a fine-tuned model later is a config change only — no app code changes.

## Error Handling

- No/too-short audio (silence, <1s): reject with a friendly message before running the
  model.
- Unsupported file formats: validate upload extension/mime type, convert via
  `pydub`/`ffmpeg`, reject if conversion fails.
- Model/dependency load failures (missing checkpoint, torch/ffmpeg not installed): fail
  loudly at startup with a clear error, not a silent crash mid-request.
- Non-Arabic speech: the model is only trained on Arabic dialects, so results on
  non-Arabic input are meaningless. No dedicated language-detection gate in v1 (YAGNI) —
  the UI will note this limitation.

## Testing

- Unit tests for audio preprocessing (resampling, format validation) using small sample
  fixtures.
- Integration test: run a couple of known-dialect sample clips through the full pipeline
  and assert the top predicted class is reasonable.
- Manual verification in the browser: record via mic, upload a file, confirm both paths
  produce results.

## Explicitly Out of Scope for v1

- Paid/cloud AI APIs of any kind.
- Natural-language explanations of *why* an accent was classified a certain way.
- Public hosting, user accounts, multi-tenant scaling.
- Language detection / non-Arabic input gating.
