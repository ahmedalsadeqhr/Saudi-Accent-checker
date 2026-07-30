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
    return max(scores.items(), key=lambda item: item[1])
