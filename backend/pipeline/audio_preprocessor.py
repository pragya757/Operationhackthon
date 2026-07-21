"""
Audio Preprocessing Service
═══════════════════════════
Applies Voice Activity Detection (VAD), noise suppression,
echo cancellation, and normalization on 16 kHz mono PCM audio streams.
"""

import numpy as np

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        # For simple echo cancellation (LMS Adaptive Filter)
        self.filter_weights = None
        self.mu = 0.01  # Step size for LMS
        
        # Try to initialize webrtcvad if available
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2) # Aggressiveness mode 2
            self.has_webrtc_vad = True
        except ImportError:
            self.has_webrtc_vad = False
            
    def apply_normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply peak normalization to the audio buffer."""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

    def apply_vad(self, audio_data: np.ndarray) -> bool:
        """
        Check if voice activity is detected in the audio chunk.
        Uses webrtcvad if available, otherwise falls back to RMS envelope detector.
        """
        if self.has_webrtc_vad:
            try:
                # webrtcvad requires 10, 20, or 30 ms frames.
                # At 16kHz, 30ms is 480 samples.
                frame_len = 480
                # Convert float32 array to 16-bit PCM bytes
                pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                # Run VAD in chunks of 480 samples
                active_frames = 0
                total_frames = 0
                for offset in range(0, len(pcm_data) - frame_len * 2 + 1, frame_len * 2):
                    frame = pcm_data[offset : offset + frame_len * 2]
                    if self.vad.is_speech(frame, self.sample_rate):
                        active_frames += 1
                    total_frames += 1
                if total_frames > 0:
                    return (active_frames / total_frames) >= 0.3
            except Exception:
                pass # Fallback to RMS
                
        # Fallback RMS energy detector
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > 0.01

    def apply_noise_suppression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Applies a spectral subtraction noise suppression filter.
        Simulates / matches RNNoise dynamics by removing continuous background noise.
        """
        if len(audio_data) < 512:
            return audio_data
            
        # FFT to spectral domain
        fft_values = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_values)
        phase = np.angle(fft_values)
        
        # Estimate noise floor dynamically using bottom 10% magnitude values
        noise_floor = np.percentile(magnitude, 15)
        
        # Apply spectral subtraction noise gates
        clean_magnitude = np.maximum(magnitude - 1.5 * noise_floor, 0.0)
        
        # Reconstruct representation
        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_audio = np.fft.irfft(clean_fft)
        
        # Ensure length matches
        if len(clean_audio) < len(audio_data):
            clean_audio = np.pad(clean_audio, (0, len(audio_data) - len(clean_audio)))
        else:
            clean_audio = clean_audio[:len(audio_data)]
            
        return clean_audio

    def apply_echo_cancellation(self, far_end: np.ndarray, near_end: np.ndarray) -> np.ndarray:
        """
        LMS Adaptive Filter for Acoustic Echo Cancellation (AEC).
        Cleans the speaker echo (far_end input feed) from the microphone (near_end input feed).
        """
        if far_end is None or len(far_end) != len(near_end):
            return near_end
            
        n = len(near_end)
        filter_len = min(64, n)
        
        if self.filter_weights is None or len(self.filter_weights) != filter_len:
            self.filter_weights = np.zeros(filter_len)
            
        cleaned = np.zeros(n)
        for i in range(filter_len, n):
            # Input window
            x = far_end[i - filter_len:i][::-1]
            # Predicted echo
            y_hat = np.dot(self.filter_weights, x)
            # Error (cleaned signal)
            err = near_end[i] - y_hat
            cleaned[i] = err
            # Weight update
            self.filter_weights += 2 * self.mu * err * x / (np.dot(x, x) + 1e-6)
            
        return cleaned

    def process(self, audio_data: np.ndarray, far_end_reference: np.ndarray = None) -> tuple[np.ndarray, bool]:
        """
        Executes entire preprocessing workflow:
        Echo cancellation → Noise suppression → Normalization → Voice Activity Check.
        """
        out = audio_data
        if far_end_reference is not None:
            out = self.apply_echo_cancellation(far_end_reference, out)
        out = self.apply_noise_suppression(out)
        out = self.apply_normalize(out)
        is_speech = self.apply_vad(out)
        return out, is_speech
