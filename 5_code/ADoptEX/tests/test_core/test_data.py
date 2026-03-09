"""Tests for core.data — spike detection, stim window, crop."""

import numpy as np
import pytest

from ADoptEX.core.data import (TraceData, crop_to_stim_window, detect_spikes,
                               find_stim_window)

# =========================================================================
# detect_spikes
# =========================================================================


class TestDetectSpikes:
    def test_flat_trace_zero_spikes(self):
        voltage = np.full(1000, -65.0)
        spikes = detect_spikes(voltage, dt_ms=0.1, threshold_mv=-20.0)
        assert len(spikes) == 0

    def test_correct_count(self):
        voltage = np.full(1000, -65.0)
        # Insert 3 spikes
        for idx in [100, 400, 700]:
            voltage[idx : idx + 3] = 20.0
        spikes = detect_spikes(voltage, dt_ms=0.1, threshold_mv=-20.0)
        assert len(spikes) == 3

    def test_ms_units(self):
        voltage = np.full(1000, -65.0)
        voltage[200:203] = 20.0
        spikes = detect_spikes(voltage, dt_ms=0.1)
        # Spike at index 200 → 200 * 0.1 = 20.0 ms
        assert spikes[0] == pytest.approx(20.0, abs=0.2)

    def test_min_interval_filtering(self):
        voltage = np.full(1000, -65.0)
        # Two spikes very close together
        voltage[100:102] = 20.0
        voltage[103:105] = 20.0
        spikes = detect_spikes(voltage, dt_ms=0.1, min_interval_ms=2.0)
        # min_interval = 2ms / 0.1ms = 20 samples → second spike filtered
        assert len(spikes) == 1

    def test_rising_edge_only(self):
        """Only rising edges should be detected."""
        voltage = np.array([-65.0, -65.0, 20.0, 20.0, 20.0, -65.0])
        spikes = detect_spikes(voltage, dt_ms=0.1, threshold_mv=-20.0)
        assert len(spikes) == 1


# =========================================================================
# find_stim_window
# =========================================================================


class TestFindStimWindow:
    def test_step_current(self):
        current = np.zeros(1000)
        current[200:800] = 500.0
        start, end = find_stim_window(current, threshold_pA=100.0)
        assert start == 200
        assert end == 799

    def test_no_stim_returns_full_range(self):
        current = np.zeros(1000)
        start, end = find_stim_window(current, threshold_pA=100.0)
        assert start == 0
        assert end == 1000

    def test_custom_threshold(self):
        current = np.full(1000, 50.0)
        current[300:600] = 200.0
        start, end = find_stim_window(current, threshold_pA=150.0)
        assert start == 300
        assert end == 599


# =========================================================================
# TraceData properties
# =========================================================================


class TestTraceData:
    @pytest.fixture
    def sample_trace(self):
        n = 1000
        dt = 0.1
        time = np.arange(n) * dt
        voltage = np.full(n, -65.0)
        voltage[200:203] = 20.0
        voltage[500:503] = 20.0
        current = np.zeros(n)
        current[100:800] = 500.0
        spike_times = np.array([20.0, 50.0])
        return TraceData(
            time=time,
            voltage=voltage,
            current=current,
            dt_ms=dt,
            spike_times=spike_times,
            stim_start_idx=100,
            stim_end_idx=800,
            stim_current_pA=500.0,
        )

    def test_n_samples(self, sample_trace):
        assert sample_trace.n_samples == 1000

    def test_duration_ms(self, sample_trace):
        assert sample_trace.duration_ms == pytest.approx(99.9)

    def test_stim_duration_ms(self, sample_trace):
        # (800 - 100) * 0.1 = 70 ms
        assert sample_trace.stim_duration_ms == pytest.approx(70.0)

    def test_n_spikes(self, sample_trace):
        assert sample_trace.n_spikes == 2

    def test_get_stim_window(self, sample_trace):
        t, v, c = sample_trace.get_stim_window()
        assert len(t) == 700
        assert len(v) == 700
        assert len(c) == 700


# =========================================================================
# crop_to_stim_window
# =========================================================================


class TestCropToStimWindow:
    @pytest.fixture
    def full_trace(self):
        n = 2000
        dt = 0.1
        time = np.arange(n) * dt
        voltage = np.full(n, -65.0)
        voltage[500:503] = 20.0
        voltage[800:803] = 20.0
        current = np.zeros(n)
        current[400:1600] = 500.0
        spike_times = np.array([50.0, 80.0])
        return TraceData(
            time=time,
            voltage=voltage,
            current=current,
            dt_ms=dt,
            spike_times=spike_times,
            stim_start_idx=400,
            stim_end_idx=1600,
            stim_current_pA=500.0,
        )

    def test_correct_boundaries_no_padding(self, full_trace):
        cropped = crop_to_stim_window(full_trace)
        assert cropped.stim_start_idx == 0
        assert cropped.stim_end_idx == 1200  # 1600 - 400
        assert cropped.n_samples == 1200

    def test_time_reset_to_zero(self, full_trace):
        cropped = crop_to_stim_window(full_trace)
        assert cropped.time[0] == pytest.approx(0.0)

    def test_max_duration_limit(self, full_trace):
        cropped = crop_to_stim_window(full_trace, max_duration_ms=50.0)
        # 50 ms / 0.1 ms = 500 samples
        assert cropped.n_samples == 500

    def test_max_duration_with_padding(self, full_trace):
        # max_duration caps the STIM window, padding is added ON TOP
        cropped = crop_to_stim_window(
            full_trace,
            max_duration_ms=50.0,
            padding_ms_before=10.0,
            padding_ms_after=10.0,
        )
        # stim: 500 samples (50ms), before: 100 (10ms), after: 100 (10ms)
        # total = 100 + 500 + 100 = 700
        assert cropped.n_samples == 700
        assert cropped.stim_start_idx == 100
        assert cropped.stim_end_idx == 600

    def test_padding_after(self, full_trace):
        cropped = crop_to_stim_window(full_trace, padding_ms_after=10.0)
        # padding_after = 10ms / 0.1ms = 100 extra samples after stim
        assert cropped.n_samples == 1300  # 1200 + 100
        assert cropped.stim_start_idx == 0
        assert cropped.stim_end_idx == 1200

    def test_padding_before(self, full_trace):
        cropped = crop_to_stim_window(full_trace, padding_ms_before=10.0)
        # padding_before = 10ms / 0.1ms = 100 extra samples before stim
        assert cropped.n_samples == 1300  # 100 + 1200
        assert cropped.stim_start_idx == 100
        assert cropped.stim_end_idx == 1300

    def test_padding_both(self, full_trace):
        cropped = crop_to_stim_window(
            full_trace, padding_ms_before=10.0, padding_ms_after=10.0
        )
        # 100 before + 1200 stim + 100 after = 1400
        assert cropped.n_samples == 1400
        assert cropped.stim_start_idx == 100
        assert cropped.stim_end_idx == 1300

    def test_padding_before_clamps_to_zero(self, full_trace):
        # Request more pre-padding than available (stim starts at idx 400 = 40 ms)
        cropped = crop_to_stim_window(full_trace, padding_ms_before=100.0)
        # Can only go back 400 samples, not 1000
        assert cropped.time[0] == pytest.approx(0.0)
        assert cropped.stim_start_idx == 400  # all 400 pre-stim samples included
        assert cropped.n_samples == 1600  # 400 + 1200

    def test_spike_time_recalculation(self, full_trace):
        cropped = crop_to_stim_window(full_trace)
        # Original spikes at 50 ms and 80 ms, stim starts at 400*0.1 = 40 ms
        # Recalculated: 50 - 40 = 10 ms, 80 - 40 = 40 ms
        assert len(cropped.spike_times) == 2
        np.testing.assert_allclose(cropped.spike_times, [10.0, 40.0], atol=0.2)

    def test_spike_times_with_padding_before(self, full_trace):
        cropped = crop_to_stim_window(full_trace, padding_ms_before=10.0)
        # crop_start = 400 - 100 = 300, time origin = 300*0.1 = 30 ms
        # Spikes at 50 and 80 ms -> 50-30=20, 80-30=50
        assert len(cropped.spike_times) == 2
        np.testing.assert_allclose(cropped.spike_times, [20.0, 50.0], atol=0.2)
