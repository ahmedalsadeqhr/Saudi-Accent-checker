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
