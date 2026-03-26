"""Tests for benchmark method definitions."""

import pytest

from ADoptEX.benchmark.methods import (
    MethodConfig,
    get_hp_sensitivity_methods,
    get_standard_methods,
    get_tonic_methods,
)


class TestGetStandardMethods:
    def test_method_count(self):
        """4 headline methods."""
        methods = get_standard_methods()
        assert len(methods) == 4

    def test_all_names_unique(self):
        """All method names are unique."""
        methods = get_standard_methods()
        names = [m.name for m in methods]
        assert len(names) == len(set(names))

    def test_method_types(self):
        """3 gradient + 1 nelder-mead."""
        methods = get_standard_methods()
        gradient = [m for m in methods if m.method_type == "gradient"]
        nm = [m for m in methods if m.method_type == "nelder_mead"]
        assert len(gradient) == 3
        assert len(nm) == 1

    def test_gradient_methods_have_config(self):
        """All gradient methods have training_config and loss_type."""
        methods = get_standard_methods()
        for m in methods:
            if m.method_type == "gradient":
                assert m.training_config is not None
                assert m.loss_type is not None

    def test_nm_method_has_settings(self):
        """Nelder-Mead method has objective and max_fev."""
        methods = get_standard_methods()
        nm = [m for m in methods if m.method_type == "nelder_mead"][0]
        assert nm.nm_max_fev > 0
        assert nm.nm_objective in ("van_rossum", "guarino", "mse")

    def test_all_use_sigmoid_transform(self):
        """All gradient methods use sigmoid transform."""
        methods = get_standard_methods()
        for m in methods:
            if m.method_type == "gradient":
                assert m.training_config.use_param_transform is True

    def test_all_return_best(self):
        """All gradient methods return best-epoch params."""
        methods = get_standard_methods()
        for m in methods:
            if m.method_type == "gradient":
                assert m.training_config.return_best is True

    def test_expected_names(self):
        """Check expected method names exist."""
        methods = get_standard_methods()
        names = {m.name for m in methods}
        assert "grad_vanrossum" in names
        assert "grad_guarino" in names
        assert "grad_mse" in names
        assert "nelder_mead" in names


class TestGetHPSensitivityMethods:
    def test_returns_list(self):
        """Returns a non-empty list of MethodConfig."""
        methods = get_hp_sensitivity_methods()
        assert len(methods) > 0
        assert all(isinstance(m, MethodConfig) for m in methods)

    def test_all_names_unique(self):
        """All HP variant names are unique."""
        methods = get_hp_sensitivity_methods()
        names = [m.name for m in methods]
        assert len(names) == len(set(names))

    def test_no_overlap_with_standard(self):
        """HP variants don't duplicate standard method names."""
        standard_names = {m.name for m in get_standard_methods()}
        hp_names = {m.name for m in get_hp_sensitivity_methods()}
        assert standard_names.isdisjoint(hp_names)

    def test_tau_variants_present(self):
        """Van Rossum tau variants are generated."""
        methods = get_hp_sensitivity_methods()
        tau_methods = [m for m in methods if "tau" in m.name]
        assert len(tau_methods) >= 2

    def test_slope_variants_present(self):
        """Surrogate slope variants are generated."""
        methods = get_hp_sensitivity_methods()
        slope_methods = [m for m in methods if "slope" in m.name]
        assert len(slope_methods) >= 2

    def test_lr_variants_present(self):
        """Learning rate variants are generated."""
        methods = get_hp_sensitivity_methods()
        lr_methods = [m for m in methods if "lr" in m.name]
        assert len(lr_methods) >= 2

    def test_adam_variant_present(self):
        """Adam optimizer variant is generated."""
        methods = get_hp_sensitivity_methods()
        adam_methods = [m for m in methods if "adam" in m.name]
        assert len(adam_methods) >= 1


class TestGetTonicMethods:
    def test_method_count(self):
        """3 methods: grad_vanrossum, grad_guarino, nelder_mead."""
        methods = get_tonic_methods()
        assert len(methods) == 3

    def test_expected_names(self):
        """Check expected method names."""
        names = {m.name for m in get_tonic_methods()}
        assert names == {"grad_vanrossum", "grad_guarino", "nelder_mead"}

    def test_method_types(self):
        """2 gradient + 1 nelder-mead."""
        methods = get_tonic_methods()
        gradient = [m for m in methods if m.method_type == "gradient"]
        nm = [m for m in methods if m.method_type == "nelder_mead"]
        assert len(gradient) == 2
        assert len(nm) == 1

    def test_gradient_epochs(self):
        """Gradient methods use 200 epochs."""
        for m in get_tonic_methods():
            if m.method_type == "gradient":
                assert m.training_config.n_epochs == 200

    def test_nm_max_fev(self):
        """NM uses 2000 max function evaluations."""
        nm = [m for m in get_tonic_methods() if m.method_type == "nelder_mead"][0]
        assert nm.nm_max_fev == 2000

    def test_surrogate_slope(self):
        """Gradient methods use surrogate_slope=2.0."""
        for m in get_tonic_methods():
            if m.method_type == "gradient":
                assert m.training_config.surrogate_slope == 2.0

    def test_all_use_polyak(self):
        """Gradient methods use Polyak optimizer."""
        for m in get_tonic_methods():
            if m.method_type == "gradient":
                assert m.training_config.optimizer == "polyak"

    def test_all_use_sigmoid_transform(self):
        """Gradient methods use sigmoid param transform."""
        for m in get_tonic_methods():
            if m.method_type == "gradient":
                assert m.training_config.use_param_transform is True

    def test_all_return_best(self):
        """Gradient methods return best-epoch params."""
        for m in get_tonic_methods():
            if m.method_type == "gradient":
                assert m.training_config.return_best is True
