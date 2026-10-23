#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Recurrent-state reset deadlines follow PCM, not processing time."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState


class RecordingModel:
    """Expose model reset positions without loading ONNX weights."""

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.reset_frames = []
        self.reset_states()

    def __call__(self, audio, sample_rate):
        self.calls += 1
        return [0.9]

    def reset_states(self):
        self.reset_frames.append(self.calls)


class TestSileroAudioClock(unittest.TestCase):
    def analyzer(self, rate):
        with patch("pipecat.audio.vad.silero.SileroOnnxModel", RecordingModel):
            analyzer = SileroVADAnalyzer()
        self.addCleanup(analyzer._executor.shutdown, wait=True)
        analyzer.set_sample_rate(rate)
        return analyzer

    def test_deadlines_use_samples_without_accumulating_frame_rounding(self):
        for rate in (8000, 16000):
            with self.subTest(rate=rate):
                analyzer = self.analyzer(rate)
                size = analyzer.num_frames_required()
                frame = np.zeros(size, dtype=np.int16).tobytes()
                period = 5 * rate
                count = math.ceil(20 * rate / size)
                # Reading wall time here is a regression, even if it is monotonic.
                with patch("time.time", side_effect=AssertionError("wall clock read")):
                    for _ in range(count):
                        self.assertEqual(analyzer.voice_confidence(frame), 0.9)
                self.assertEqual(
                    analyzer._model.reset_frames,
                    [0] + [math.ceil(k * period / size) for k in range(1, 5)],
                )
                self.assertEqual(analyzer._samples_since_reset, count * size % period)

    def test_first_frame_does_not_discard_recurrent_context(self):
        analyzer = self.analyzer(16000)
        analyzer.voice_confidence(bytes(1024))
        self.assertEqual(analyzer._model.reset_frames, [0])
        self.assertEqual(analyzer._samples_since_reset, 512)

    def test_model_reset_preserves_gate_volume_and_candidate_counts(self):
        analyzer = self.analyzer(16000)
        size = analyzer.num_frames_required()
        for _ in range(math.ceil(5 * 16000 / size) - 1):
            analyzer.voice_confidence(bytes(size * 2))
        analyzer._vad_state = VADState.SPEAKING
        analyzer._prev_volume = 0.8
        analyzer._vad_starting_count = 3
        analyzer._vad_stopping_count = 2
        analyzer.voice_confidence(bytes(size * 2))
        self.assertEqual(len(analyzer._model.reset_frames), 2)
        self.assertEqual(analyzer._vad_state, VADState.SPEAKING)
        self.assertEqual(analyzer._prev_volume, 0.8)
        self.assertEqual(analyzer._vad_starting_count, 3)
        self.assertEqual(analyzer._vad_stopping_count, 2)

    def test_chunk_sizes_do_not_move_model_reset_positions(self):
        for rate in (8000, 16000):
            expected = None
            for chunk_ms in (10, 20, 40):
                with self.subTest(rate=rate, chunk_ms=chunk_ms):
                    analyzer = self.analyzer(rate)
                    pcm = bytes(11 * rate * 2)
                    size = rate * chunk_ms // 1000 * 2
                    for offset in range(0, len(pcm), size):
                        analyzer._run_analyzer(pcm[offset : offset + size])
                    positions = analyzer._model.reset_frames
                    if expected is None:
                        expected = positions
                    self.assertEqual(positions, expected)

    def test_tracks_and_new_streams_have_independent_audio_clocks(self):
        user = self.analyzer(16000)
        assistant = self.analyzer(16000)
        for _ in range(math.ceil(5 * 16000 / 512)):
            user.voice_confidence(bytes(1024))
        assistant.voice_confidence(bytes(1024))
        fresh = self.analyzer(16000)
        self.assertEqual(len(user._model.reset_frames), 2)
        self.assertEqual(assistant._model.reset_frames, [0])
        self.assertEqual(assistant._samples_since_reset, 512)
        self.assertEqual(fresh._samples_since_reset, 0)
        self.assertEqual(fresh._model.reset_frames, [0])

    def test_sample_rate_change_restarts_sample_budget(self):
        analyzer = self.analyzer(16000)
        analyzer.voice_confidence(bytes(1024))
        analyzer.set_sample_rate(8000)
        self.assertEqual(analyzer._samples_since_reset, 0)
        for _ in range(math.ceil(5 * 8000 / 256)):
            analyzer.voice_confidence(bytes(512))
        self.assertEqual(len(analyzer._model.reset_frames), 2)


if __name__ == "__main__":
    unittest.main()
