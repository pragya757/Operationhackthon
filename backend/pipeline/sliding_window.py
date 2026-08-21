"""
Sliding Window Stream Manager
═════════════════════════════
Maintains a sliding buffer of audio samples.
1.5-second rolling windows with 0.5-second step updates
for fast real-time transcript and score output (~1.5s first result).
"""

import numpy as np

class SlidingWindowBuffer:
    def __init__(self, window_size_sec: float = 1.5, step_size_sec: float = 0.5, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.window_samples = int(window_size_sec * sample_rate)  # 24,000 samples @ 1.5s
        self.step_samples   = int(step_size_sec   * sample_rate)  # 8,000 samples @ 0.5s
        self.buffer = np.zeros(0, dtype=np.float32)
        
        # Track total processed steps to yield exact timestamp alignments
        self.steps_processed = 0

    def append_samples(self, new_samples: np.ndarray):
        """Append fresh PCM samples to the rolling buffer."""
        self.buffer = np.concatenate((self.buffer, new_samples))

    def has_next_window(self) -> bool:
        """
        Check if we have enough accumulated samples to extract
        the next step boundary.
        """
        required_len = self.window_samples + (self.steps_processed * self.step_samples)
        # However, if we just started, we wait until we compile the first 3 seconds (48,000 samples)
        if len(self.buffer) < self.window_samples:
            return False
            
        # If we have already started, we check if we have enough samples for the next 1-second step.
        current_offset = self.steps_processed * self.step_samples
        return len(self.buffer) >= (current_offset + self.window_samples)

    def get_next_window(self) -> tuple[np.ndarray, float, float]:
        """
        Get the next window of 3 seconds.
        Returns:
            window: 3 seconds of float32 samples (48,000 values)
            start_sec: float timestamp of start of window
            end_sec: float timestamp of end of window
        """
        if not self.has_next_window():
            return None, 0.0, 0.0
            
        start_idx = self.steps_processed * self.step_samples
        end_idx = start_idx + self.window_samples
        
        window = self.buffer[start_idx:end_idx]
        
        start_sec = start_idx / self.sample_rate
        end_sec = end_idx / self.sample_rate
        
        self.steps_processed += 1
        
        # Maintain buffer size: garbage collect samples that are older than twice the window size
        max_retained_samples = self.window_samples * 2
        discard_limit = start_idx - max_retained_samples
        if discard_limit > 0:
            self.buffer = self.buffer[discard_limit:]
            self.steps_processed -= discard_limit // self.step_samples
            
        return window, start_sec, end_sec

    def reset(self):
        """Clear active session buffers."""
        self.buffer = np.zeros(0, dtype=np.float32)
        self.steps_processed = 0
