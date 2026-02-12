"""Tests for loss.guarino — hard detection and experimental feature extraction."""

import jax.numpy as jnp
import numpy as np
import pytest

from ADoptEX.loss.guarino import (GuarinoFeatures, detect_spikes_hard,
                                  extract_experimental_features)

# =========================================================================
# detect_spikes_hard
# =========================================================================


class TestDetectSpikesHard:
    def test_no_spikes(self):
        voltage = jnp.full(1000, -65.0)
        times = detect_spikes_hard(voltage, threshold_mv=0.0, dt_ms=0.1)
        assert len(times) == 0

    def test_correct_count(self):
        v = np.full(1000, -65.0)
        for idx in [100, 400, 700]:
            v[idx : idx + 3] = 20.0
        times = detect_spikes_hard(jnp.array(v), threshold_mv=0.0, dt_ms=0.1)
        assert len(times) == 3

    def test_correct_times(self):
        v = np.full(1000, -65.0)
        v[200:203] = 20.0
        times = detect_spikes_hard(jnp.array(v), threshold_mv=0.0, dt_ms=0.1)
        assert float(times[0]) == pytest.approx(20.0, abs=0.2)

    def test_min_interval(self):
        v = np.full(1000, -65.0)
        v[100:102] = 20.0
        v[103:105] = 20.0  # Only 3 samples apart = 0.3 ms
        times = detect_spikes_hard(
            jnp.array(v), threshold_mv=0.0, min_interval_ms=2.0, dt_ms=0.1
        )
        assert len(times) == 1

    def test_threshold_sensitivity(self):
        v = np.full(1000, -65.0)
        v[200:203] = -10.0  # Below 0 mV threshold
        times_strict = detect_spikes_hard(jnp.array(v), threshold_mv=0.0, dt_ms=0.1)
        times_loose = detect_spikes_hard(jnp.array(v), threshold_mv=-20.0, dt_ms=0.1)
        assert len(times_strict) == 0
        assert len(times_loose) == 1


# =========================================================================
# extract_experimental_features
# =========================================================================


class TestExtractExperimentalFeatures:
    def _make_voltage_with_spikes(self, spike_indices):
        v = np.full(2000, -65.0)
        for idx in spike_indices:
            v[idx : idx + 5] = 20.0
        return jnp.array(v)

    def test_correct_type(self):
        v = self._make_voltage_with_spikes([200, 600, 1000])
        feats = extract_experimental_features(v, dt_ms=0.1, stim_duration_ms=200.0)
        assert isinstance(feats, GuarinoFeatures)

    def test_spike_count(self):
        v = self._make_voltage_with_spikes([200, 600, 1000])
        feats = extract_experimental_features(v, dt_ms=0.1, stim_duration_ms=200.0)
        assert float(feats.n_spikes) == 3.0

    def test_binary_validity_flags(self):
        v = self._make_voltage_with_spikes([200, 600, 1000])
        feats = extract_experimental_features(v, dt_ms=0.1, stim_duration_ms=200.0)
        assert float(feats.has_first_spike) == 1.0
        assert float(feats.has_second_spike) == 1.0
        assert float(feats.has_third_spike) == 1.0

    def test_no_spike_features(self):
        v = jnp.full(2000, -65.0)
        feats = extract_experimental_features(v, dt_ms=0.1, stim_duration_ms=200.0)
        assert float(feats.n_spikes) == 0.0
        assert float(feats.has_first_spike) == 0.0
        assert float(feats.t_first_spike) == 0.0

    def test_isi_values(self):
        # Spikes at 20 ms and 60 ms → ISI = 40 ms → inv = 1/40
        v = self._make_voltage_with_spikes([200, 600])
        feats = extract_experimental_features(v, dt_ms=0.1, stim_duration_ms=200.0)
        assert float(feats.inv_first_isi) == pytest.approx(1.0 / 40.0, rel=0.1)
