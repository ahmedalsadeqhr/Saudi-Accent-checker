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
