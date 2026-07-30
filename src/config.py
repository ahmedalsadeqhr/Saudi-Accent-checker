TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SEC = 1.0

DEFAULT_MODEL_SOURCE = "Elyadata/ADI-whisper-ADI17"
MODEL_SAVE_DIR = "pretrained_models/dialect_id"

# ADI17 label conventions vary in case/format across model releases; match
# any of these against a normalized (upper, stripped) label string.
SAUDI_LABEL_ALIASES = frozenset({"KSA", "SAU", "SA"})
