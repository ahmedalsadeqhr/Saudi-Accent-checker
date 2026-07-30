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
