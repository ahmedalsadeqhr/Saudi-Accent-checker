TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SEC = 1.0

DEFAULT_MODEL_SOURCE = "badrex/mms-300m-arabic-dialect-identifier"

# This model's label set groups Saudi in with the broader "Gulf" class.
# Match case-insensitively against a normalized (upper, stripped) label string.
SAUDI_LABEL_ALIASES = frozenset({"GULF"})
