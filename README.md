# Saudi Accent Checker

Record or upload Arabic speech and see confidence scores across Arabic dialects,
with the Saudi/Gulf match highlighted. Runs fully locally — no paid APIs.

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) installed and on
your PATH (needed for non-WAV file uploads).

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the printed local URL (and the "Network URL" to let teammates on the same
network access it).

The first prediction downloads the pretrained model (`badrex/mms-300m-arabic-dialect-identifier`)
into the local Hugging Face cache — this can take a few minutes and needs
internet access once, then works offline.

## Tests

```bash
pytest -m "not integration"   # fast unit tests
pytest -m integration         # slow, downloads/runs the real model
```

## Fine-tuning later

To use your own fine-tuned checkpoint instead of the stock pretrained model,
point `src/config.py`'s `DEFAULT_MODEL_SOURCE` at your checkpoint's local path
or Hugging Face repo — no other code changes required.

## Accuracy evaluation

To measure the deployed model's accuracy against labeled ground truth from the public
`ArabicSpeech/ADI17` dataset (weighted toward the Saudi/Gulf case):

```bash
pip install requests  # if not already installed
python scripts/evaluate_accuracy.py
```

This fetches ~75 labeled clips over the network (no local dataset download), classifies
each with the current model, and writes a report to `docs/eval/accuracy-report-<date>.md`
with overall accuracy, per-dialect accuracy, and a confusion matrix. Note: Modern Standard
Arabic (MSA) isn't evaluated against ground truth — the source dataset has no MSA samples.
