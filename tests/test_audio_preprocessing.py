import io

import numpy as np
import pytest
import soundfile as sf

from src.audio_preprocessing import (
    AudioTooShortError,
    UnsupportedAudioError,
    load_audio,
    prepare_audio,
    resample,
    to_mono,
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
