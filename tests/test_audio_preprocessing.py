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
