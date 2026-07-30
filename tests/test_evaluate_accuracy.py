from unittest.mock import Mock, patch

import pytest
import requests

from scripts.evaluate_accuracy import (
    MAX_PAGE_RETRIES,
    _download_audio_with_retry,
    _get_page_with_retry,
    _get_with_retry,
    find_labeled_clip_urls,
)


def _fake_response(status_code: int, json_data=None, content=b""):
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.content = content
    response.json.return_value = json_data if json_data is not None else {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error"
        )
    else:
        response.raise_for_status.side_effect = None
    return response


@pytest.fixture(autouse=True)
def no_real_sleep():
    with patch("scripts.evaluate_accuracy.time.sleep", return_value=None):
        yield


def test_429_response_triggers_a_retry():
    responses = [_fake_response(429), _fake_response(200, json_data={"ok": True})]
    with patch("scripts.evaluate_accuracy.requests.get", side_effect=responses) as mock_get:
        result = _get_with_retry("https://example.test/rows")
    assert result.json() == {"ok": True}
    assert mock_get.call_count == 2


def test_non_retryable_4xx_raises_immediately_without_retrying():
    with patch("scripts.evaluate_accuracy.requests.get", return_value=_fake_response(404)) as mock_get:
        with pytest.raises(requests.exceptions.HTTPError):
            _get_with_retry("https://example.test/missing")
    assert mock_get.call_count == 1


def test_exhausting_retries_raises_instead_of_returning_none():
    with patch(
        "scripts.evaluate_accuracy.requests.get",
        return_value=_fake_response(503),
    ) as mock_get:
        with pytest.raises(requests.exceptions.HTTPError):
            _get_with_retry("https://example.test/rows")
    assert mock_get.call_count == MAX_PAGE_RETRIES


def test_connection_errors_are_retried_then_raised():
    with patch(
        "scripts.evaluate_accuracy.requests.get",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ) as mock_get:
        with pytest.raises(requests.exceptions.ConnectionError):
            _get_with_retry("https://example.test/rows")
    assert mock_get.call_count == MAX_PAGE_RETRIES


def test_get_page_with_retry_returns_parsed_json():
    with patch(
        "scripts.evaluate_accuracy.requests.get",
        return_value=_fake_response(200, json_data={"rows": []}),
    ):
        result = _get_page_with_retry({"offset": 0})
    assert result == {"rows": []}


def test_download_audio_with_retry_returns_content():
    with patch(
        "scripts.evaluate_accuracy.requests.get",
        return_value=_fake_response(200, content=b"fake-audio-bytes"),
    ):
        result = _download_audio_with_retry("https://example.test/clip.wav")
    assert result == b"fake-audio-bytes"


def _row(dialect_code: str, src: str):
    return {"row": {"dialect": dialect_code, "audio": [{"src": src}]}}


def test_find_labeled_clip_urls_stops_once_sample_plan_is_satisfied():
    # SAMPLE_PLAN = {"KSA": 20, "UAE": 10, "EGY": 15, "JOR": 15, "MOR": 15}; use a small
    # patched plan so the test doesn't need 75 fake rows.
    small_plan = {"KSA": 2, "EGY": 1}
    page_one = {
        "rows": [
            _row("KSA", "https://example.test/ksa1.wav"),
            _row("EGY", "https://example.test/egy1.wav"),
            _row("UAE", "https://example.test/uae1.wav"),  # not in plan, ignored
        ],
        "num_rows_total": 1000,
    }
    page_two = {
        "rows": [
            _row("KSA", "https://example.test/ksa2.wav"),
        ],
        "num_rows_total": 1000,
    }
    responses = [
        _fake_response(200, json_data=page_one),
        _fake_response(200, json_data=page_two),
    ]
    with patch("scripts.evaluate_accuracy.SAMPLE_PLAN", small_plan):
        with patch("scripts.evaluate_accuracy.requests.get", side_effect=responses) as mock_get:
            found = find_labeled_clip_urls()

    assert mock_get.call_count == 2
    assert found == [
        ("KSA", "https://example.test/ksa1.wav"),
        ("EGY", "https://example.test/egy1.wav"),
        ("KSA", "https://example.test/ksa2.wav"),
    ]


def test_find_labeled_clip_urls_stops_when_no_more_rows_available():
    small_plan = {"KSA": 5}
    page_one = {
        "rows": [_row("KSA", "https://example.test/ksa1.wav")],
        "num_rows_total": 1,
    }
    with patch("scripts.evaluate_accuracy.SAMPLE_PLAN", small_plan):
        with patch(
            "scripts.evaluate_accuracy.requests.get",
            return_value=_fake_response(200, json_data=page_one),
        ) as mock_get:
            found = find_labeled_clip_urls()

    assert mock_get.call_count == 1
    assert found == [("KSA", "https://example.test/ksa1.wav")]
