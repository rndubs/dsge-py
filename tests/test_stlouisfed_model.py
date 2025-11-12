"""
Tests for the St. Louis Fed DSGE model.

Tests cover model creation, parameter specification, matrix dimensions,
and basic solution properties.
"""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stlouisfed_dsge import StLouisFedDSGE, create_stlouisfed_dsge


class TestStLouisFedModelCreation:
    """Test model instantiation and basic properties."""

    def test_model_creation(self):
        """Test that model can be created successfully."""
        model = create_stlouisfed_dsge()
        assert isinstance(model, StLouisFedDSGE)
        assert model.spec is not None

    def test_model_dimensions(self):
        """Test that model dimensions are correct."""
        model = create_stlouisfed_dsge()

        # Check dimensions
        assert model.spec.n_states == 49  # ~49 states
        assert model.spec.n_controls == 0  # No controls (all states)
        assert model.spec.n_shocks == 7  # 7 structural shocks
        assert model.spec.n_observables == 13  # 13 observables

    def test_state_names(self):
        """Test that state names are properly defined."""
        model = create_stlouisfed_dsge()

        # Check that key states exist
        assert "C" in model.spec.state_names  # Aggregate consumption
        assert "CH" in model.spec.state_names  # Worker consumption
        assert "CS" in model.spec.state_names  # Capitalist consumption
        assert "Y" in model.spec.state_names  # Output
        assert "I" in model.spec.state_names  # Investment
        assert "G" in model.spec.state_names  # Government spending
        assert "B" in model.spec.state_names  # Government debt
        assert "tax" in model.spec.state_names  # Taxes
        assert "H" in model.spec.state_names  # Hours
        assert "W" in model.spec.state_names  # Wages
        assert "PIE" in model.spec.state_names  # Inflation

    def test_shock_names(self):
        """Test that shock names are properly defined."""
        model = create_stlouisfed_dsge()

        expected_shocks = ["epsZ", "epsM", "epsG", "epsMS", "epsWMS", "epsPr", "epsZI"]
        assert model.spec.shock_names == expected_shocks

    def test_observable_names(self):
        """Test that observable names are properly defined."""
        model = create_stlouisfed_dsge()

        expected_observables = [
            "dy",  # GDP growth
            "dc",  # Consumption growth
            "dinve",  # Investment growth
            "dg",  # Government spending growth
            "hours",  # Hours worked
            "dw",  # Wage growth
            "infl",  # Inflation
            "ffr",  # Federal Funds Rate
            "r10y",  # 10-Year Treasury Rate
            "infl_exp",  # Inflation expectations
            "ls",  # Labor share
            "debt_gdp",  # Debt/GDP ratio
            "tax_gdp",  # Tax/GDP ratio
        ]
        assert model.spec.observable_names == expected_observables


class TestStLouisFedParameters:
    """Test parameter definitions and priors."""

    def test_parameter_count(self):
        """Test that correct number of parameters are defined."""
        model = create_stlouisfed_dsge()

        # Should have ~40 parameters
        assert len(model.parameters) > 35
        assert len(model.parameters) < 50

    def test_key_parameters_exist(self):
        """Test that key structural parameters exist."""
        model = create_stlouisfed_dsge()
        params = model.parameters._params

        # Preference parameters
        assert "betta" in params
        assert "sigma_c" in params
        assert "varrho" in params

        # Production parameters
        assert "alp" in params
        assert "delta" in params

        # Heterogeneity parameters
        assert "lambda_w" in params  # Worker share
        assert "psiH" in params  # Portfolio adjustment cost

        # Policy parameters
        assert "rho_r" in params  # Interest rate smoothing
        assert "theta_pie" in params  # Taylor rule inflation
        assert "phi_tauT_B" in params  # Fiscal rule debt response

        # Shock persistence
        assert "rhoZ" in params
        assert "rhoG" in params

        # Shock standard deviations
        assert "sigma_Z" in params
        assert "sigma_G" in params

    def test_parameter_bounds(self):
        """Test that parameters have reasonable bounds."""
        model = create_stlouisfed_dsge()

        # Discount factor
        betta = model.parameters.get("betta")
        assert betta.bounds[0] > 0.8
        assert betta.bounds[1] < 1.0

        # Worker share
        lambda_w = model.parameters.get("lambda_w")
        assert lambda_w.bounds[0] >= 0.0
        assert lambda_w.bounds[1] <= 1.0

    def test_prior_specifications(self):
        """Test that parameters have appropriate priors."""
        model = create_stlouisfed_dsge()

        # Check that estimable parameters have priors
        betta = model.parameters.get("betta")
        assert betta.prior is not None
        assert betta.prior.distribution == "beta"

        # Check that fixed parameters don't require priors
        Hss = model.parameters.get("Hss")
        assert Hss.fixed is True

    def test_steady_state_values(self):
        """Test steady state parameter values."""
        model = create_stlouisfed_dsge()

        # Key steady state values from Cantore-Freund calibration
        assert model.parameters.get("Hss").value == pytest.approx(0.33, rel=0.01)
        assert model.parameters.get("gy").value == pytest.approx(0.20, rel=0.01)
        assert model.parameters.get("LSss").value == pytest.approx(0.67, rel=0.01)


class TestStLouisFedSystemMatrices:
    """Test system matrix construction."""

    def test_system_matrices_return_correct_structure(self):
        """Test that system_matrices returns expected dictionary."""
        model = create_stlouisfed_dsge()
        matrices = model.system_matrices()

        assert isinstance(matrices, dict)
        assert "Gamma0" in matrices
        assert "Gamma1" in matrices
        assert "Psi" in matrices
        assert "Pi" in matrices

    def test_system_matrices_dimensions(self):
        """Test that system matrices have correct dimensions."""
        model = create_stlouisfed_dsge()
        matrices = model.system_matrices()

        n_states = model.spec.n_states
        n_shocks = model.spec.n_shocks

        # Check Gamma matrices
        assert matrices["Gamma0"].shape == (n_states, n_states)
        assert matrices["Gamma1"].shape == (n_states, n_states)

        # Check Psi matrix (shock loadings)
        assert matrices["Psi"].shape == (n_states, n_shocks)

        # Check Pi matrix (expectation errors)
        assert matrices["Pi"].shape[0] == n_states

    def test_system_matrices_no_nans(self):
        """Test that system matrices don't contain NaN values."""
        model = create_stlouisfed_dsge()
        matrices = model.system_matrices()

        assert not np.any(np.isnan(matrices["Gamma0"]))
        assert not np.any(np.isnan(matrices["Gamma1"]))
        assert not np.any(np.isnan(matrices["Psi"]))
        assert not np.any(np.isnan(matrices["Pi"]))

    def test_shock_loadings(self):
        """Test that shock loadings are present in Psi matrix."""
        model = create_stlouisfed_dsge()
        matrices = model.system_matrices()

        # Psi matrix should have non-zero entries for shocks
        Psi = matrices["Psi"]
        assert np.any(Psi != 0.0), "Psi matrix should have non-zero shock loadings"


class TestStLouisFedMeasurementEquation:
    """Test measurement equation construction."""

    def test_measurement_equation_dimensions(self):
        """Test that measurement matrices have correct dimensions."""
        model = create_stlouisfed_dsge()
        Z, D = model.measurement_equation()

        n_obs = model.spec.n_observables
        n_states = model.spec.n_states

        assert Z.shape == (n_obs, n_states)
        assert D.shape == (n_obs,)

    def test_measurement_equation_no_nans(self):
        """Test that measurement matrices don't contain NaN values."""
        model = create_stlouisfed_dsge()
        Z, D = model.measurement_equation()

        assert not np.any(np.isnan(Z))
        assert not np.any(np.isnan(D))

    def test_observable_mappings(self):
        """Test that observables map to appropriate states."""
        model = create_stlouisfed_dsge()
        Z, D = model.measurement_equation()

        # Get indices
        obs_names = model.spec.observable_names
        state_names = model.spec.state_names

        # Check that key observables have non-zero mappings
        for obs_name in obs_names:
            obs_idx = obs_names.index(obs_name)
            row = Z[obs_idx, :]
            assert np.any(row != 0.0), f"Observable {obs_name} should map to at least one state"


class TestStLouisFedModelValidation:
    """Test model validation and consistency checks."""

    def test_model_validation(self):
        """Test that model passes internal validation."""
        model = create_stlouisfed_dsge()

        # Model should validate successfully
        # (This will be implemented in the base class)
        # For now, just check that it doesn't raise an exception
        try:
            matrices = model.system_matrices()
            Z, D = model.measurement_equation()
            assert True
        except Exception as e:
            pytest.fail(f"Model validation failed: {str(e)}")

    def test_heterogeneity_features(self):
        """Test that model includes heterogeneity features."""
        model = create_stlouisfed_dsge()

        # Check that worker and capitalist variables exist
        assert "CH" in model.spec.state_names  # Worker consumption
        assert "CS" in model.spec.state_names  # Capitalist consumption
        assert "BH" in model.spec.state_names  # Worker bonds
        assert "BS" in model.spec.state_names  # Capitalist bonds

        # Check heterogeneity parameters
        assert "lambda_w" in model.parameters._params  # Population share
        assert "psiH" in model.parameters._params  # Portfolio adjustment cost

    def test_fiscal_sector(self):
        """Test that model includes fiscal sector."""
        model = create_stlouisfed_dsge()

        # Check fiscal variables
        assert "G" in model.spec.state_names  # Government spending
        assert "B" in model.spec.state_names  # Government debt
        assert "tax" in model.spec.state_names  # Taxes

        # Check fiscal parameters
        assert "phi_tauT_B" in model.parameters._params  # Tax response to debt
        assert "phi_tauT_G" in model.parameters._params  # Tax response to spending
        assert "gy" in model.parameters._params  # Spending/GDP ratio
        assert "BYss" in model.parameters._params  # Debt/GDP ratio


@pytest.mark.skip(reason="Model requires full equilibrium system implementation for solution")
class TestStLouisFedModelSolution:
    """Tests for model solution (requires complete implementation)."""

    def test_model_solution(self):
        """Test that model can be solved (placeholder for future)."""
        # This test will be enabled once full equilibrium system is implemented
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
