import numpy as np
import scipy.signal


class SignalReliabilityEstimator:
    """
    Evaluate the reliability (confidence) of HR and HRV estimates.
    Combines multiple metrics: spectral quality, temporal stability, etc.
    """

    def __init__(self, fs: float = 30.0):
        self.fs = fs

    def spectral_quality(self, signal: np.ndarray) -> float:
        """Compute spectral band energy ratio (heartband vs total)."""
        if len(signal) < 60:
            return np.nan
        f, pxx = scipy.signal.welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
        mask = (f >= 0.7) & (f <= 1.8) # heart rate band in Hz
        band_power = np.trapz(pxx[mask], f[mask])
        total_power = np.trapz(pxx, f)
        return band_power / total_power if total_power > 0 else np.nan

    def hr_stability(self, hr_values: np.ndarray) -> float:
        """Estimate temporal HR stability: 1 - normalized variance."""
        if len(hr_values) < 10:
            return np.nan
        cv = np.std(hr_values) / np.mean(hr_values)  # coefficient de variation
        return np.clip(1 - cv, 0, 1)

    def combine(self, signal: np.ndarray, hr_values: np.ndarray) -> float:
        """Combine metrics into a single reliability score [0, 1]."""
        sq = self.spectral_quality(signal)
        st = self.hr_stability(hr_values)
        if not np.isfinite(sq) or not np.isfinite(st):
            return np.nan
        return 0.6 * sq + 0.4 * st  # pondération empirique
    
    def reliability_series(
        self, signal: np.ndarray, hr_values: np.ndarray
    ) -> np.ndarray:
        """Compute reliability per sample (rolling window)."""
        n = len(signal)
        if n < 60:
            return np.full(n, np.nan)
        win = min(200, n // 4)
        rel = np.zeros(n)
        for i in range(n):
            start = max(0, i - win // 2)
            end = min(n, i + win // 2)
            rel[i] = self.combine(signal[start:end], hr_values[max(0, len(hr_values) - (end - start)):])
        return rel

