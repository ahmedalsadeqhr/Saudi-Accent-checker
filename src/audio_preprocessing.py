import numpy as np
import torchaudio
import torch


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
