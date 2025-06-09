# utils/augment.py

from __future__ import annotations

import math
import random
from typing import Dict, Any, Optional

import numpy as np
import scipy.signal
import librosa


# ------------------------------------------------------------------ #
# Default hyper-parameters (override by passing cfg=dict(...)).
# ------------------------------------------------------------------ #
_DEFAULTS: Dict[str, Any] = {
    # Waveform-level
    "prob_speed":      0.30,
    "speed_range":     (0.90, 1.10),  # rate < 1 ⇒ slower & lower pitch
    "prob_noise":      0.30,
    "snr_db":          (10, 25),      # higher = cleaner
    "prob_reverb":     0.20,
    "rir_scale":       0.03,          # 30 ms synthetic impulse
    "prob_gain":       1.00,          # always apply mild gain jitter
    "gain_range":      (0.8, 1.25),

    # SpecAugment
    "prob_specaugment": 0.50,
    "T":                50,   # max time-mask width (frames)
    "F":                15,   # max freq-mask width (mel bins)
    "num_masks":        2,    # how many independent (T,F) masks
}

# ------------------------------------------------------------------ #
# Helper: Merge user overrides with defaults.
# ------------------------------------------------------------------ #
def _get_cfg(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = _DEFAULTS.copy()
    if user:
        cfg.update(user)
    return cfg


# ------------------------------------------------------------------ #
# --------------------   W A V E F O R M   -------------------------- #
# ------------------------------------------------------------------ #
def augment_wave(wav: np.ndarray, sr: int, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Randomly augments a mono waveform in-memory.

    Operations (each toggled by probability):
      • Speed & pitch (time-stretch)
      • Additive white noise at random SNR
      • Synthetic room reverb (FFT-convolution with unit impulse)
      • Loudness / gain jitter

    The ordering is fixed to avoid overly destructive combos.
    """
    cfg = _get_cfg(cfg)

    # ----- Speed / pitch -----------------------------------------------------
    if random.random() < cfg["prob_speed"]:
        rate = random.uniform(*cfg["speed_range"])
        # librosa.time_stretch works on STFT; for short clips use resample
        wav = librosa.effects.time_stretch(wav, rate)
        # Restore original length by resampling back to sr
        wav = librosa.resample(wav, orig_sr=int(sr / rate), target_sr=sr)

    # ----- Additive Gaussian noise ------------------------------------------
    if random.random() < cfg["prob_noise"]:
        snr_db = random.uniform(*cfg["snr_db"])
        # Power ratio
        noise_power = np.mean(wav ** 2) / (10 ** (snr_db / 10))
        noise = np.random.normal(0, math.sqrt(noise_power), wav.shape)
        wav = wav + noise

    # ----- Simple room reverb -----------------------------------------------
    if random.random() < cfg["prob_reverb"]:
        rir_len = int(cfg["rir_scale"] * sr)
        rir = np.zeros(rir_len, dtype=np.float32)
        rir[0] = 1.0  # direct path
        # Exponentially decaying tail with tiny random bumps
        decay = np.exp(-np.linspace(0, 3, rir_len))
        rir += 0.3 * decay * np.random.randn(rir_len)
        wav = scipy.signal.fftconvolve(wav, rir, mode="full")[: len(wav)]

    # ----- Gain / loudness ---------------------------------------------------
    if random.random() < cfg["prob_gain"]:
        gain = random.uniform(*cfg["gain_range"])
        wav = wav * gain

    # Safety clip
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav


# ------------------------------------------------------------------ #
# ------------------   S P E C A U G M E N T   --------------------- #
# ------------------------------------------------------------------ #
def specaugment(mel: np.ndarray, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Basic SpecAugment (time- & frequency-masking only).

    The implementation is deliberately simple and CPU-only because the
    Real-Time-Voice-Cloning repo does mel augmentation on-the-fly
    inside Python `Dataset` objects.

    Parameters
    ----------
    mel : 2-D ndarray [n_mels, T]
    cfg : optional dict with keys T, F, num_masks, prob_specaugment

    Returns
    -------
    Augmented mel-spectrogram (np.float32).
    """
    cfg = _get_cfg(cfg)
    if random.random() >= cfg["prob_specaugment"]:
        return mel  # untouched

    m = mel.copy()

    n_mels, n_frames = m.shape
    for _ in range(cfg["num_masks"]):
        # -- Time mask --------------------------------------------------------
        t = random.randint(0, cfg["T"])
        t0 = random.randint(0, max(1, n_frames - t))
        m[:, t0 : t0 + t] = 0.0

        # -- Frequency mask ---------------------------------------------------
        f = random.randint(0, cfg["F"])
        f0 = random.randint(0, max(1, n_mels - f))
        m[f0 : f0 + f, :] = 0.0

    return m.astype(np.float32)


# ------------------------------------------------------------------ #
# -----------------------   M I S C   ------------------------------ #
# ------------------------------------------------------------------ #
__all__ = ["augment_wave", "specaugment"]

if __name__ == "__main__":
    # Quick smoke-test (not executed during import).
    import soundfile as sf

    test_wav, sr = librosa.load(librosa.ex("trumpet"), sr=None)
    print("Original:", test_wav.shape, test_wav.dtype, sr)

    aug = augment_wave(test_wav, sr)
    print("Augmented:", aug.shape, aug.dtype)

    # Save to disk for manual listening
    sf.write("orig.wav", test_wav, sr)
    sf.write("aug.wav",  aug,       sr)
    print("➡️  Wrote orig.wav / aug.wav")
