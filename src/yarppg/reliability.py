import numpy as np
import scipy.signal


class SignalReliabilityEstimator:
    """
    Evaluate the reliability (confidence) of HR and HRV estimates.
    Combines multiple metrics: spectral quality, temporal stability, etc.
    """

    def __init__(self, fs: float = 30.0):
        self.fs = fs

    def spectral_quality(self, signal: np.ndarray, hr_values: np.ndarray) -> float:
        """Compute improved SNR-based spectral quality.
        
        Uses peak detection and adaptive bandwidth for better signal/noise separation.
        """
        if len(signal) < 60:
            return np.nan
            
        # Compute power spectral density
        f, pxx = scipy.signal.welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
        
        # Define heart rate frequency range (0.5-3.0 Hz = 30-180 BPM)
        hr_mask = (f >= 0.5) & (f <= 3.0)
        hr_freqs = f[hr_mask]
        hr_power = pxx[hr_mask]
        
        if len(hr_freqs) < 3:
            return np.nan
            
        # Find the dominant peak in heart rate range
        peaks, properties = scipy.signal.find_peaks(hr_power, height=np.mean(hr_power))
        
        if len(peaks) == 0:
            return 0.0  # No clear peak found
            
        # Find the most prominent peak
        peak_prominences = scipy.signal.peak_prominences(hr_power, peaks)[0]
        dominant_peak_idx = peaks[np.argmax(peak_prominences)]
        peak_freq = hr_freqs[dominant_peak_idx]
        peak_power = hr_power[dominant_peak_idx]
        
        # Use actual HR if available to validate peak
        if len(hr_values) > 0 and np.isfinite(hr_values[-1]):
            expected_freq = hr_values[-1] / 60.0  # Convert BPM to Hz
            freq_error = abs(peak_freq - expected_freq)
            if freq_error > 0.2:  # Peak too far from expected HR
                return 0.0
        
        # Define signal band around the peak (±0.1 Hz)
        signal_band_halfwidth = 0.1
        signal_mask = (f >= peak_freq - signal_band_halfwidth) & (f <= peak_freq + signal_band_halfwidth)
        
        # Calculate signal power
        signal_power = np.trapz(pxx[signal_mask], f[signal_mask])
        
        # Define noise regions
        # 1. Adjacent noise: frequencies just outside signal band
        adjacent_mask = ((f >= peak_freq - 0.3) & (f < peak_freq - signal_band_halfwidth)) | \
                       ((f > peak_freq + signal_band_halfwidth) & (f <= peak_freq + 0.3))
        
        # 2. Baseline noise: average power in non-peak regions
        baseline_mask = (f >= 0.1) & (f <= 4.0) & ~signal_mask & ~adjacent_mask
        
        # Calculate noise power
        adjacent_power = np.trapz(pxx[adjacent_mask], f[adjacent_mask]) if np.any(adjacent_mask) else 0
        baseline_power = np.mean(pxx[baseline_mask]) * (f[-1] - f[0]) if np.any(baseline_mask) else 0
        noise_power = adjacent_power + baseline_power
        
        # Calculate SNR
        if noise_power == 0:
            return 1.0 if signal_power > 0 else 0.0
            
        snr = signal_power / noise_power
        
        # Normalize SNR to 0-1 range (log scale)
        # Good SNR is typically > 10 (20 dB), excellent > 100 (40 dB)
        snr_score = np.clip(np.log10(snr + 1) / 3.0, 0.0, 1.0)  # log10(100+1) ≈ 2, so /3 gives ~0.67
        
        # Apply peak quality penalties
        # 1. Peak prominence penalty
        prominence_score = np.clip(peak_prominences[np.argmax(peak_prominences)] / np.mean(hr_power), 0.0, 1.0)
        
        # 2. Peak width penalty (narrower peaks are better)
        peak_width = 0.2  # Default width
        if len(peaks) > 0:
            # Estimate peak width at half maximum
            peak_height = peak_power
            half_max = peak_height / 2
            left_idx = np.where(hr_power[:dominant_peak_idx] <= half_max)[0]
            right_idx = np.where(hr_power[dominant_peak_idx:] <= half_max)[0]
            if len(left_idx) > 0 and len(right_idx) > 0:
                width_samples = right_idx[0] + dominant_peak_idx - left_idx[-1]
                peak_width = width_samples * (f[1] - f[0])  # Convert to Hz
        
        width_score = np.clip(1 - peak_width / 0.5, 0.0, 1.0)  # Penalty for wide peaks
        
        # Combine SNR with peak quality
        final_score = snr_score * 0.6 + prominence_score * 0.25 + width_score * 0.15
        
        return float(np.clip(final_score, 0.0, 1.0))

    def hr_stability(self, hr_values: np.ndarray) -> float:
        """Estimate temporal HR stability: 1 - normalized variance."""
        if len(hr_values) < 10:
            return np.nan
        cv = np.std(hr_values) / np.mean(hr_values)  # coefficient de variation
        return np.clip(1 - cv, 0, 1)

    def compute_metrics(
        self, signal: np.ndarray, hr_values: np.ndarray
    ) -> dict[str, float]:
        """Return individual reliability metrics along with the combined score."""
        sq = float(self.spectral_quality(signal, hr_values))
        st = float(self.hr_stability(hr_values))
        morph = float(self.morphology_quality(signal, hr_values))

        components: list[tuple[float, float]] = []
        if np.isfinite(sq):
            components.append((0.4, sq)) #0.4
        if np.isfinite(st):
            components.append((0.3, st)) #0.3
        if np.isfinite(morph):
            components.append((0.3, morph)) #0.3

        if not components:
            combined = np.nan
        else:
            weights, values = zip(*components)
            weights = np.asarray(weights, dtype=float).ravel()
            values = np.asarray(values, dtype=float).ravel()
            weights /= weights.sum()
            combined = float(np.dot(weights, values))

        return {
            "reliability": combined,
            "spectral_quality": sq,
            "hr_stability": st,
            "morphology_quality": morph,
        }

    def combine(self, signal: np.ndarray, hr_values: np.ndarray) -> float:
        """Combine metrics into a single reliability score [0, 1]."""
        return self.compute_metrics(signal, hr_values)["reliability"]
    
    def reliability_series(
        self, signal: np.ndarray, hr_values: np.ndarray
    ) -> np.ndarray:
        """Compute reliability per sample (rolling window)."""
        n = len(signal)
        if n < 60:
            return np.full(n, np.nan)
        win = max(60, min(200, n))
        rel = np.zeros(n)
        for i in range(n):
            start = max(0, i - win // 2)
            end = start + win
            if end > n:
                end = n
                start = max(0, end - win)
            window_signal = signal[start:end]
            window_len = len(window_signal)
            if window_len < 60:
                rel[i] = np.nan
                continue

            if len(hr_values) >= window_len:
                window_hr = hr_values[-window_len:]
            else:
                window_hr = hr_values

            rel[i] = self.combine(window_signal, window_hr)

        return rel

    def morphology_quality(self, signal: np.ndarray, hr_values: np.ndarray) -> float:
        """Estimate signal morphology quality using filtered PPG signal.
        
        Designed to work with bandpass-filtered signals (0.5-2 Hz).
        Focuses on features preserved by the filter: pulse shape, regularity, and timing.
        """
        sig = np.asarray(signal, dtype=float)
        sig = sig[np.isfinite(sig)]
        min_samples = int(max(self.fs * 2, 60))
        if len(sig) < min_samples:
            return np.nan

        # Remove baseline (median centering)
        sig = sig - np.nanmedian(sig)
        
        # Apply Savitzky-Golay filter for smoothing (preserves shape better than moving average)
        window = int(max(5, int(self.fs * 0.6)))
        if window % 2 == 0:
            window += 1
        if window > len(sig):
            window = len(sig) if len(sig) % 2 == 1 else len(sig) - 1
        if window >= 5 and window < len(sig):
            sig = scipy.signal.savgol_filter(sig, window_length=window, polyorder=3, mode="interp")

        # === COMPONENT 1: PULSE SHAPE REGULARITY (40% weight) ===
        # Analyze the consistency of pulse wave shapes
        d1 = np.gradient(sig)
        
        # Calculate pulse-to-pulse variability
        peaks, _ = scipy.signal.find_peaks(sig, distance=int(self.fs * 0.4))  # Min 0.4s between peaks
        if len(peaks) < 3:
            shape_score = 0.0
        else:
            # Analyze peak-to-peak intervals
            peak_intervals = np.diff(peaks)
            interval_cv = np.std(peak_intervals) / np.mean(peak_intervals)  # Coefficient of variation
            shape_score = np.clip(1 - interval_cv, 0.0, 1.0)  # Lower CV = more regular = higher score
        
        # === COMPONENT 2: UPSTROKE/DOWNSTROKE BALANCE (35% weight) ===
        # This is preserved by the bandpass filter
        pos_energy = np.sum(np.square(d1[d1 > 0]))
        neg_energy = np.sum(np.square(d1[d1 < 0]))
        total_energy = pos_energy + abs(neg_energy)
        if total_energy == 0:
            slope_score = 0.0
        else:
            upstroke_ratio = pos_energy / total_energy
            # Ideal ratio for PPG is around 0.55 (slightly more upstroke energy)
            slope_score = 1 - abs(upstroke_ratio - 0.55) / 0.55
            slope_score = np.clip(slope_score, 0.0, 1.0)
        
        # === COMPONENT 3: SIGNAL SMOOTHNESS (25% weight) ===
        # Analyze how smooth the signal is (low noise, good filtering)
        d2 = np.gradient(d1)
        signal_energy = np.mean(np.square(sig))
        noise_energy = np.mean(np.square(d2))  # High 2nd derivative = noise/irregularities
        
        if signal_energy == 0:
            smoothness_score = 0.0
        else:
            noise_ratio = noise_energy / signal_energy
            # Lower noise ratio = smoother signal = higher score
            smoothness_score = np.clip(1 - noise_ratio * 10, 0.0, 1.0)  # Scale factor may need tuning
        
        # === FINAL SCORE ===
        return float(0.4 * shape_score + 0.35 * slope_score + 0.25 * smoothness_score)
