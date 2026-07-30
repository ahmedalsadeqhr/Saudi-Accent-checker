# Saudi Accent Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app where a user records or uploads Arabic speech and gets back confidence scores per Arabic dialect, with the Saudi/Gulf (KSA) class highlighted — using a free, local, pretrained model (no paid APIs).

**Architecture:** A single Streamlit app (`app.py`) calls into two small pure-Python modules: `src/audio_preprocessing.py` (validates and normalizes incoming audio) and `src/dialect_classifier.py` (wraps a pretrained `transformers` audio-classification model and returns per-dialect scores). Config constants (model source, sample rate, thresholds) live in `src/config.py` so the model can be swapped for a fine-tuned checkpoint later without touching app code.

**Tech Stack:** Python 3.10+, Streamlit, Hugging Face `transformers`, PyTorch/torchaudio, soundfile, pydub (+ system ffmpeg), pytest.

## Global Constraints

- No paid/cloud APIs — all inference runs locally.
- Target sample rate for the model: 16000 Hz, mono (from spec).
- Minimum accepted audio duration: 1.0 second (from spec) — shorter/silent input is rejected before running the model.
- Pretrained model: `badrex/mms-300m-arabic-dialect-identifier` (`transformers` `audio-classification` pipeline, MMS-300m fine-tuned on 5 Arabic varieties: Maghrebi, MSA, Egyptian, Gulf, Levantine — Saudi is represented by the `Gulf` label). Model source is a config value, not hardcoded inline, so it can be swapped for a fine-tuned checkpoint later.
  - **Deviation note:** the plan originally pinned `Elyadata/ADI-whisper-ADI17` (SpeechBrain). That model turned out to require custom SpeechBrain interface code cloned from a separate GitHub repo (not bundled on Hugging Face) plus a ~2.5-5GB Whisper-large-v3 checkpoint. Discovered during Task 3 implementation and swapped, with the human partner's approval, for the lighter model above, which works with the standard `transformers` pipeline (no custom code, no git clone, ~0.3B params, CC BY 4.0 licensed).
- No natural-language explanation of *why* — label + confidence scores only (out of scope per spec).
- No language-detection gate for non-Arabic input (out of scope per spec) — note the limitation in the UI copy only.

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `pytest.ini`

**Interfaces:**
- Produces: `src.config.TARGET_SAMPLE_RATE: int`, `src.config.MIN_AUDIO_DURATION_SEC: float`, `src.config.DEFAULT_MODEL_SOURCE: str`, `src.config.SAUDI_LABEL_ALIASES: frozenset[str]`

> **Superseded by the Task 3 model-swap deviation (see Global Constraints):** as originally executed, this task installed `speechbrain` instead of `transformers`, pinned `DEFAULT_MODEL_SOURCE = "Elyadata/ADI-whisper-ADI17"`, defined a now-unused `MODEL_SAVE_DIR`, and set `SAUDI_LABEL_ALIASES = frozenset({"KSA", "SAU", "SA"})`. Task 3 updates `requirements.txt` and `src/config.py` to the values below as part of its own commits — this section is left as a historical record of what Task 1 actually did; the blocks below show the corrected target state.

- [ ] **Step 1: Create `requirements.txt`**

```text
streamlit>=1.38
transformers>=4.40
torch>=2.2
torchaudio>=2.2
soundfile>=0.12
pydub>=0.25
audioop-lts>=0.2.1
numpy>=1.26
pytest>=8.0
```

- [ ] **Step 2: Create `src/__init__.py`** (empty file, makes `src` a package)

- [ ] **Step 3: Create `src/config.py`**

```python
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SEC = 1.0

DEFAULT_MODEL_SOURCE = "badrex/mms-300m-arabic-dialect-identifier"

# This model's label set groups Saudi in with the broader "Gulf" class.
# Match case-insensitively against a normalized (upper, stripped) label string.
SAUDI_LABEL_ALIASES = frozenset({"GULF"})
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    integration: slow tests that download/load the real pretrained model
```

- [ ] **Step 5: Install dependencies and verify imports**

Run: `pip install -r requirements.txt && python -c "import streamlit, transformers, torch, torchaudio, soundfile, pydub; from src import config; print(config.TARGET_SAMPLE_RATE)"`
Expected: prints `16000` with no import errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini src/__init__.py src/config.py
git commit -m "chore: scaffold project structure and config"
```

---

### Task 2: Audio preprocessing module

**Files:**
- Create: `src/audio_preprocessing.py`
- Test: `tests/test_audio_preprocessing.py`

**Interfaces:**
- Consumes: `src.config.TARGET_SAMPLE_RATE`, `src.config.MIN_AUDIO_DURATION_SEC`
- Produces:
  - `src.audio_preprocessing.AudioTooShortError(Exception)`
  - `src.audio_preprocessing.UnsupportedAudioError(Exception)`
  - `src.audio_preprocessing.to_mono(waveform: np.ndarray) -> np.ndarray` — averages channels if `waveform.ndim == 2`, passes through 1-D arrays unchanged.
  - `src.audio_preprocessing.resample(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray`
  - `src.audio_preprocessing.validate_duration(waveform: np.ndarray, sample_rate: int, min_duration_sec: float = MIN_AUDIO_DURATION_SEC) -> None` — raises `AudioTooShortError` if too short.
  - `src.audio_preprocessing.load_audio(file_path_or_buffer) -> tuple[np.ndarray, int]` — reads via `soundfile`, falls back to `pydub` (which shells out to ffmpeg) for formats soundfile can't read (e.g. mp3/m4a); raises `UnsupportedAudioError` if neither can decode it. Returns `(waveform, sample_rate)`.
  - `src.audio_preprocessing.prepare_audio(file_path_or_buffer) -> np.ndarray` — composes `load_audio` → `to_mono` → `resample` → `validate_duration`, returns the final mono 16kHz waveform ready for the classifier.

- [ ] **Step 1: Write failing tests for `to_mono` and `resample`**

```python
# tests/test_audio_preprocessing.py
import numpy as np
import pytest

from src.audio_preprocessing import (
    AudioTooShortError,
    UnsupportedAudioError,
    to_mono,
    resample,
    validate_duration,
)


def test_to_mono_averages_stereo_channels():
    stereo = np.array([[1.0, 3.0], [0.0, 1.0]]).T  # shape (2, 2): left/right rows
    stereo = np.array([[1.0, 0.0], [3.0, 1.0]])  # 2 channels x 2 samples
    result = to_mono(stereo)
    assert result.shape == (2,)
    assert np.allclose(result, [2.0, 0.5])


def test_to_mono_passes_through_mono_unchanged():
    mono = np.array([0.1, 0.2, 0.3])
    result = to_mono(mono)
    assert np.array_equal(result, mono)


def test_resample_changes_length_proportionally():
    original = np.sin(np.linspace(0, 2 * np.pi, 8000))  # 8000 samples at 8000 Hz = 1 second
    resampled = resample(original, orig_sr=8000, target_sr=16000)
    assert resampled.shape[0] == 16000


def test_resample_noop_when_same_rate():
    original = np.array([0.1, 0.2, 0.3])
    resampled = resample(original, orig_sr=16000, target_sr=16000)
    assert np.array_equal(resampled, original)


def test_validate_duration_raises_when_too_short():
    short_waveform = np.zeros(4000)  # 0.25s at 16000 Hz
    with pytest.raises(AudioTooShortError):
        validate_duration(short_waveform, sample_rate=16000, min_duration_sec=1.0)


def test_validate_duration_passes_when_long_enough():
    long_waveform = np.zeros(16000)  # exactly 1.0s at 16000 Hz
    validate_duration(long_waveform, sample_rate=16000, min_duration_sec=1.0)  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.audio_preprocessing'`

- [ ] **Step 3: Implement `to_mono`, `resample`, `validate_duration`, and the error classes**

```python
# src/audio_preprocessing.py
import numpy as np
import torchaudio
import torch


class AudioTooShortError(Exception):
    """Raised when audio is shorter than the minimum required duration."""


class UnsupportedAudioError(Exception):
    """Raised when the audio file/buffer cannot be decoded by any supported backend."""


def to_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=0)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def validate_duration(waveform: np.ndarray, sample_rate: int, min_duration_sec: float) -> None:
    duration_sec = waveform.shape[0] / sample_rate
    if duration_sec < min_duration_sec:
        raise AudioTooShortError(
            f"Audio is {duration_sec:.2f}s, shorter than the required {min_duration_sec}s minimum."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_preprocessing.py -v`
Expected: the 6 tests written so far all PASS.

- [ ] **Step 5: Write failing tests for `load_audio` and `prepare_audio`**

```python
# append to tests/test_audio_preprocessing.py
import io
import soundfile as sf

from src.audio_preprocessing import load_audio, prepare_audio


def _write_wav_bytes(samples: np.ndarray, sample_rate: int) -> io.BytesIO:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


def test_load_audio_reads_wav_buffer():
    samples = np.sin(np.linspace(0, 2 * np.pi, 8000)).astype(np.float32)
    buffer = _write_wav_bytes(samples, sample_rate=8000)

    waveform, sample_rate = load_audio(buffer)

    assert sample_rate == 8000
    assert waveform.shape[0] == 8000


def test_load_audio_raises_on_garbage_bytes():
    garbage = io.BytesIO(b"not an audio file at all")
    with pytest.raises(UnsupportedAudioError):
        load_audio(garbage)


def test_prepare_audio_returns_mono_16k_waveform_long_enough():
    two_seconds_at_8k = np.sin(np.linspace(0, 4 * np.pi, 16000)).astype(np.float32)
    buffer = _write_wav_bytes(two_seconds_at_8k, sample_rate=8000)

    result = prepare_audio(buffer)

    assert result.ndim == 1
    assert result.shape[0] == 32000  # 2s resampled to 16000 Hz


def test_prepare_audio_raises_audio_too_short_error():
    half_second_at_16k = np.zeros(8000, dtype=np.float32)
    buffer = _write_wav_bytes(half_second_at_16k, sample_rate=16000)

    with pytest.raises(AudioTooShortError):
        prepare_audio(buffer)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_audio_preprocessing.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_audio'` (and `prepare_audio`).

- [ ] **Step 7: Implement `load_audio` and `prepare_audio`**

```python
# append to src/audio_preprocessing.py
from pydub import AudioSegment
import soundfile as sf

from src.config import TARGET_SAMPLE_RATE, MIN_AUDIO_DURATION_SEC


def load_audio(file_path_or_buffer) -> tuple[np.ndarray, int]:
    try:
        waveform, sample_rate = sf.read(file_path_or_buffer, dtype="float32")
        return waveform.T if waveform.ndim > 1 else waveform, sample_rate
    except Exception:
        pass

    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        segment = AudioSegment.from_file(file_path_or_buffer)
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        samples /= float(1 << (8 * segment.sample_width - 1))
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels)).T
        return samples, segment.frame_rate
    except Exception as exc:
        raise UnsupportedAudioError(f"Could not decode audio: {exc}") from exc


def prepare_audio(file_path_or_buffer) -> np.ndarray:
    waveform, sample_rate = load_audio(file_path_or_buffer)
    mono = to_mono(waveform)
    resampled = resample(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    validate_duration(resampled, sample_rate=TARGET_SAMPLE_RATE, min_duration_sec=MIN_AUDIO_DURATION_SEC)
    return resampled
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_audio_preprocessing.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/audio_preprocessing.py tests/test_audio_preprocessing.py
git commit -m "feat: add audio preprocessing (mono, resample, duration validation, decode)"
```

---

### Task 3: Dialect classifier wrapper

**Files:**
- Modify: `requirements.txt` (replace `speechbrain>=1.0` with `transformers>=4.40`, add `audioop-lts>=0.2.1` if not already present)
- Modify: `src/config.py` (update `DEFAULT_MODEL_SOURCE` and `SAUDI_LABEL_ALIASES`, remove `MODEL_SAVE_DIR`)
- Create: `src/dialect_classifier.py`
- Test: `tests/test_dialect_classifier.py`

**Interfaces:**
- Consumes: `src.config.DEFAULT_MODEL_SOURCE`, `src.config.SAUDI_LABEL_ALIASES`, `src.audio_preprocessing.prepare_audio`
- Produces:
  - `src.dialect_classifier.DialectClassifier(model_source: str = DEFAULT_MODEL_SOURCE)` — class with:
    - `.predict(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, float]` — dialect label → probability, sorted descending by probability.
    - `.is_saudi_label(label: str) -> bool` — normalizes and checks against `SAUDI_LABEL_ALIASES`.
  - `src.dialect_classifier.top_result(scores: dict[str, float]) -> tuple[str, float]` — returns the `(label, probability)` pair with the highest probability.

**Model note:** uses `badrex/mms-300m-arabic-dialect-identifier` via the standard `transformers` `pipeline("audio-classification", ...)` API — no custom inference code, no git clone. Its 5 output labels are `Maghrebi`, `MSA`, `Egyptian`, `Gulf`, `Levantine`; Saudi Arabic falls under `Gulf`.

- [ ] **Step 0: Update `requirements.txt` and `src/config.py` to the corrected model**

In `requirements.txt`, replace the `speechbrain>=1.0` line with `transformers>=4.40`, and add `audioop-lts>=0.2.1` if it isn't already present from Task 1's fix round. Rewrite `src/config.py` to:

```python
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SEC = 1.0

DEFAULT_MODEL_SOURCE = "badrex/mms-300m-arabic-dialect-identifier"

# This model's label set groups Saudi in with the broader "Gulf" class.
# Match case-insensitively against a normalized (upper, stripped) label string.
SAUDI_LABEL_ALIASES = frozenset({"GULF"})
```

Run `python -c "import transformers; from src import config; print(config.DEFAULT_MODEL_SOURCE)"` and confirm it prints the model name with no import errors. Commit this alone first:

```bash
git add requirements.txt src/config.py
git commit -m "fix: switch to badrex/mms-300m-arabic-dialect-identifier (transformers pipeline)"
```

- [ ] **Step 1: Write failing unit tests using a fake underlying pipeline (no download)**

These tests verify the wrapper's scoring/formatting logic without needing the real
model download, by injecting a fake `transformers` pipeline callable.

```python
# tests/test_dialect_classifier.py
import numpy as np
import pytest

from src.dialect_classifier import DialectClassifier, top_result


def _fake_pipeline(audio_input, top_k=None):
    return [
        {"label": "Gulf", "score": 0.7},
        {"label": "Egyptian", "score": 0.2},
        {"label": "Levantine", "score": 0.1},
    ]


@pytest.fixture
def classifier():
    instance = DialectClassifier.__new__(DialectClassifier)
    instance._pipe = _fake_pipeline
    return instance


def test_predict_returns_label_to_probability_mapping(classifier):
    waveform = np.zeros(16000, dtype=np.float32)
    scores = classifier.predict(waveform)
    assert scores == {"Gulf": pytest.approx(0.7), "Egyptian": pytest.approx(0.2), "Levantine": pytest.approx(0.1)}


def test_predict_is_sorted_descending_by_probability(classifier):
    waveform = np.zeros(16000, dtype=np.float32)
    scores = classifier.predict(waveform)
    assert list(scores.keys()) == ["Gulf", "Egyptian", "Levantine"]


def test_is_saudi_label_matches_gulf_case_insensitively(classifier):
    assert classifier.is_saudi_label("Gulf") is True
    assert classifier.is_saudi_label("GULF") is True
    assert classifier.is_saudi_label("Egyptian") is False


def test_top_result_returns_highest_probability_pair():
    scores = {"Gulf": 0.7, "Egyptian": 0.2, "Levantine": 0.1}
    assert top_result(scores) == ("Gulf", 0.7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dialect_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dialect_classifier'`

- [ ] **Step 3: Implement `DialectClassifier` and `top_result`**

```python
# src/dialect_classifier.py
import numpy as np
from transformers import pipeline

from src.config import DEFAULT_MODEL_SOURCE, SAUDI_LABEL_ALIASES, TARGET_SAMPLE_RATE


class DialectClassifier:
    def __init__(self, model_source: str = DEFAULT_MODEL_SOURCE):
        self._pipe = pipeline("audio-classification", model=model_source)

    def predict(self, waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, float]:
        results = self._pipe({"array": waveform, "sampling_rate": sample_rate}, top_k=None)
        scores = {result["label"]: result["score"] for result in results}
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def is_saudi_label(self, label: str) -> bool:
        return label.strip().upper() in SAUDI_LABEL_ALIASES


def top_result(scores: dict) -> tuple:
    return next(iter(scores.items()))
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `pytest tests/test_dialect_classifier.py -v -m "not integration"`
Expected: the 4 fake-pipeline tests PASS.

- [ ] **Step 5: Write an integration test against the real pretrained model**

```python
# append to tests/test_dialect_classifier.py

@pytest.mark.integration
def test_real_model_loads_and_predicts_on_silence():
    """Downloads the real pretrained model on first run (slow, needs internet)."""
    classifier = DialectClassifier()
    waveform = np.zeros(16000, dtype=np.float32)
    scores = classifier.predict(waveform)
    assert len(scores) > 0
    assert all(isinstance(p, float) for p in scores.values())
    label, probability = top_result(scores)
    assert isinstance(label, str)
    assert 0.0 <= probability <= 1.0
```

- [ ] **Step 6: Run the integration test to verify it passes**

Run: `pytest tests/test_dialect_classifier.py -v -m integration`
Expected: PASS (first run downloads the ~0.3B-parameter model via the `transformers`/Hugging Face cache; this can take a few minutes).

- [ ] **Step 7: Commit**

```bash
git add src/dialect_classifier.py tests/test_dialect_classifier.py
git commit -m "feat: add dialect classifier wrapper around pretrained transformers pipeline"
```

---

### Task 4: Streamlit app

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `src.audio_preprocessing.prepare_audio`, `src.audio_preprocessing.AudioTooShortError`, `src.audio_preprocessing.UnsupportedAudioError`, `src.dialect_classifier.DialectClassifier`, `src.dialect_classifier.top_result`, `src.dialect_classifier.SAUDI_LABEL_ALIASES` (via `classifier.is_saudi_label`)
- Produces: a runnable Streamlit page (no importable interface consumed by later tasks — this is the last task).

- [ ] **Step 1: Write `app.py`**

```python
# app.py
import streamlit as st

from src.audio_preprocessing import AudioTooShortError, UnsupportedAudioError, prepare_audio
from src.dialect_classifier import DialectClassifier, top_result

st.set_page_config(page_title="Saudi Accent Checker", page_icon="\U0001F3A4")
st.title("Saudi Accent Checker")
st.caption(
    "Record or upload Arabic speech to see confidence scores across Arabic dialects. "
    "Results are meaningless for non-Arabic speech — there is no language filter yet."
)


@st.cache_resource
def get_classifier() -> DialectClassifier:
    return DialectClassifier()


def render_scores(scores: dict) -> None:
    label, probability = top_result(scores)
    classifier = get_classifier()
    if classifier.is_saudi_label(label):
        st.success(f"Top match: **{label}** (Saudi/Gulf) — {probability:.0%} confidence")
    else:
        st.info(f"Top match: **{label}** — {probability:.0%} confidence (not Saudi/Gulf)")
    st.bar_chart(scores)


tab_record, tab_upload = st.tabs(["Record", "Upload a file"])

audio_input = None
with tab_record:
    recorded = st.audio_input("Record yourself speaking Arabic")
    if recorded is not None:
        audio_input = recorded

with tab_upload:
    uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded is not None:
        audio_input = uploaded

if audio_input is not None:
    try:
        waveform = prepare_audio(audio_input)
    except AudioTooShortError:
        st.error("That clip is too short — please provide at least 1 second of speech.")
    except UnsupportedAudioError:
        st.error("Couldn't read that audio file. Try a WAV, MP3, or M4A file.")
    else:
        with st.spinner("Analyzing accent..."):
            scores = get_classifier().predict(waveform)
        render_scores(scores)
```

- [ ] **Step 2: Manually verify the record path**

Run: `streamlit run app.py`, open the app in a browser, click the "Record" tab, record ~2 seconds of Arabic speech, stop recording.
Expected: a bar chart of dialect scores appears with a top-match message (green if Saudi/Gulf, blue info otherwise).

- [ ] **Step 3: Manually verify the upload path**

In the same running app, switch to the "Upload a file" tab and upload a short WAV/MP3 clip of Arabic speech.
Expected: same bar chart + top-match message behavior as the record path.

- [ ] **Step 4: Manually verify error handling**

Upload a silent/near-empty file shorter than 1 second, then upload a non-audio file (e.g. a `.txt` renamed to `.wav`).
Expected: "too short" error for the first, "couldn't read that audio file" error for the second — no stack trace shown to the user.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI for recording/uploading and viewing dialect scores"
```

---

### Task 5: README and run instructions

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only)
- Produces: nothing (documentation only)

- [ ] **Step 1: Write `README.md`**

```markdown
# Saudi Accent Checker

Record or upload Arabic speech and see confidence scores across Arabic dialects,
with the Saudi/Gulf match highlighted. Runs fully locally — no paid APIs.

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) installed and on
your PATH (needed for non-WAV file uploads).

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Run

\`\`\`bash
streamlit run app.py
\`\`\`

Open the printed local URL (and the "Network URL" to let teammates on the same
network access it).

The first prediction downloads the pretrained model (`badrex/mms-300m-arabic-dialect-identifier`)
into the local Hugging Face cache — this can take a few minutes and needs
internet access once, then works offline.

## Tests

\`\`\`bash
pytest -m "not integration"   # fast unit tests
pytest -m integration         # slow, downloads/runs the real model
\`\`\`

## Fine-tuning later

To use your own fine-tuned checkpoint instead of the stock pretrained model,
point `src/config.py`'s `DEFAULT_MODEL_SOURCE` at your checkpoint's local path
or Hugging Face repo — no other code changes required.
```

- [ ] **Step 2: Verify the README's commands work as documented**

Run: `pip install -r requirements.txt` (in a clean virtualenv if possible) and `pytest -m "not integration"`.
Expected: dependencies install cleanly and unit tests pass, matching what the README claims.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add setup, run, and test instructions"
```

---

## Self-Review Notes

- **Spec coverage:** live mic + upload (Task 4), detailed-but-scores-only report (Tasks 3-4), Streamlit hosting (Task 4), free local model with a swappable `MODEL_SOURCE` for future fine-tuning (Tasks 1, 3), error handling for short/silent/unsupported audio (Tasks 2, 4), unit + integration tests (Tasks 2, 3) — all covered.
- **Type consistency:** `prepare_audio` returns `np.ndarray` used directly by `DialectClassifier.predict`; `predict` returns `dict[str, float]` used by both `top_result` and `render_scores`; `is_saudi_label` takes the same `label: str` type returned by `top_result`. Confirmed consistent across Tasks 2-4.
- **Out of scope confirmed absent:** no paid API calls, no generated explanation text, no language-detection gate, no user accounts/hosting infra anywhere in the plan.
