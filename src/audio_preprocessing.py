import numpy as np
import torchaudio
import torch
from pydub import AudioSegment
import soundfile as sf

from src.config import TARGET_SAMPLE_RATE, MIN_AUDIO_DURATION_SEC


class AudioTooShortError(Exception):
    """Raised when audio is shorter than the minimum required duration."""


class UnsupportedAudioError(Exception):
    """Raised when the audio file/buffer cannot be decoded by any supported backend."""


def to_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=0)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def validate_duration(waveform: np.ndarray, sample_rate: int, min_duration_sec: float) -> None:
    duration_sec = waveform.shape[0] / sample_rate
    if duration_sec < min_duration_sec:
        raise AudioTooShortError(
            f"Audio is {duration_sec:.2f}s, shorter than the required {min_duration_sec}s minimum."
        )


def load_audio(file_path_or_buffer) -> tuple[np.ndarray, int]:
    if hasattr(file_path_or_buffer, "seek"):
        file_path_or_buffer.seek(0)

    sf_error: Exception | None = None
    try:
        waveform, sample_rate = sf.read(file_path_or_buffer, dtype="float32")
        return waveform.T if waveform.ndim > 1 else waveform, sample_rate
    except Exception as sf_exc:
        sf_error = sf_exc

    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        segment = AudioSegment.from_file(file_path_or_buffer)
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        samples /= float(1 << (8 * segment.sample_width - 1))
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels)).T
        return samples, segment.frame_rate
    except Exception as exc:
        raise UnsupportedAudioError(f"Could not decode audio (soundfile: {sf_error}; pydub: {exc})") from exc


def prepare_audio(file_path_or_buffer) -> np.ndarray:
    waveform, sample_rate = load_audio(file_path_or_buffer)
    mono = to_mono(waveform)
    resampled = resample(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    validate_duration(resampled, sample_rate=TARGET_SAMPLE_RATE, min_duration_sec=MIN_AUDIO_DURATION_SEC)
    return resampled
