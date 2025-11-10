"""Tests for the Smets-Wouters (2007) DSGE model implementation."""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.smets_wouters_2007 import create_smets_wouters_model
from src.dsge.solvers.linear import solve_linear_model


class TestSmetsWouters2007Model:
    """Test suite for Smets-Wouters (2007) model."""

    @pytest.fixture
    def model(self):
        """Create a Smets-Wouters model instance."""
        return create_smets_wouters_model()

    def test_model_creation(self, model) -> None:
        """Test that the model can be created with correct dimensions."""
        assert model.spec.n_states == 41
        assert model.spec.n_controls == 0
        assert model.spec.n_shocks == 7
        assert model.spec.n_observables == 7
        assert len(model.parameters) == 41

    def test_state_names(self, model) -> None:
        """Test that state names are correctly defined."""
        expected_states = [
            # Sticky price (13)
            "c",
            "inve",
            "y",
            "lab",
            "k",
            "pk",
            "zcap",
            "rk",
            "w",
            "r",
            "pinf",
            "mc",
            "kp",
            # Flexible price (11)
            "cf",
            "invef",
            "yf",
            "labf",
            "kf",
            "pkf",
            "zcapf",
            "rkf",
            "wf",
            "rrf",
            "kpf",
            # Lags (8)
            "c_lag",
            "inve_lag",
            "y_lag",
            "w_lag",
            "r_lag",
            "pinf_lag",
            "kp_lag",
            "kpf_lag",
            # Shocks (7)
            "a",
            "b",
            "g",
            "qs",
            "ms",
            "spinf",
            "sw",
            # MA lags (2)
            "epinfma_lag",
            "ewma_lag",
        ]
        assert model.spec.state_names == expected_states

    def test_shock_names(self, model) -> None:
        """Test that shock names are correctly defined."""
        expected_shocks = ["ea", "eb", "eg", "eqs", "em", "epinf", "ew"]
        assert model.spec.shock_names == expected_shocks

    def test_observable_names(self, model) -> None:
        """Test that observable names are correctly defined."""
        expected_obs = [
            "obs_dy",
            "obs_dc",
            "obs_dinve",
            "obs_dw",
            "obs_pinfobs",
            "obs_robs",
            "obs_labobs",
        ]
        assert model.spec.observable_names == expected_obs

    def test_parameter_count(self, model) -> None:
        """Test that all parameters are defined."""
        # Should have structural params + shock std devs + fixed params
        assert len(model.parameters) == 41

        # Check some key parameters exist
        param_dict = model.parameters.to_dict()
        assert "csigma" in param_dict
        assert "chabb" in param_dict
        assert "calfa" in param_dict
        assert "cprobp" in param_dict
        assert "cprobw" in param_dict
        assert "crpi" in param_dict

    def test_system_matrices_shape(self, model) -> None:
        """Test that system matrices have correct dimensions."""
        mats = model.system_matrices()

        assert "Gamma0" in mats
        assert "Gamma1" in mats
        assert "Psi" in mats
        assert "Pi" in mats

        assert mats["Gamma0"].shape == (41, 41)
        assert mats["Gamma1"].shape == (41, 41)
        assert mats["Psi"].shape == (41, 7)
        assert mats["Pi"].shape == (41, 13)  # Number of expectation errors

    def test_system_matrices_not_all_zeros(self, model) -> None:
        """Test that system matrices are not trivially zero."""
        mats = model.system_matrices()

        assert np.any(mats["Gamma0"] != 0)
        assert np.any(mats["Gamma1"] != 0)
        assert np.any(mats["Psi"] != 0)
        # Pi may be sparse or zero depending on expectations

    def test_measurement_equation_shape(self, model) -> None:
        """Test that measurement matrices have correct dimensions."""
        Z, D = model.measurement_equation()

        assert Z.shape == (7, 41)  # 7 observables, 41 states
        assert D.shape == (7,)

    def test_measurement_equation_not_trivial(self, model) -> None:
        """Test that measurement equation is not trivially zero."""
        Z, _D = model.measurement_equation()

        assert np.any(Z != 0)
        # D may be zero or non-zero depending on steady state

    def test_steady_state(self, model) -> None:
        """Test that steady state is computed."""
        ss = model.steady_state()

        assert ss.shape == (41,)
        # For log-linearized model, steady state should be all zeros
        assert np.allclose(ss, 0.0)

    def test_solve_model(self, model) -> None:
        """Test that the model can be solved."""
        mats = model.system_matrices()

        try:
            solution, _info = solve_linear_model(
                Gamma0=mats["Gamma0"],
                Gamma1=mats["Gamma1"],
                Psi=mats["Psi"],
                Pi=mats["Pi"],
                n_states=model.spec.n_states,
            )

            # Check that solution has required attributes
            assert hasattr(solution, "T")
            assert hasattr(solution, "R")
            assert hasattr(solution, "C")

            # Check dimensions
            n_states = model.spec.n_states
            n_shocks = model.spec.n_shocks

            assert solution.T.shape == (n_states, n_states)
            assert solution.R.shape == (n_states, n_shocks)
            assert solution.C.shape == (n_states,)

            # Check eigenvalues for stability
            eigenvalues = np.linalg.eigvals(solution.T)
            max_eigenvalue = np.max(np.abs(eigenvalues))

            # Model should be stable (eigenvalues < 1.1 allowing for near-unit roots)
            assert max_eigenvalue < 1.1, f"Model unstable: max eigenvalue = {max_eigenvalue}"

        except Exception as e:
            pytest.fail(f"Model solution failed: {e}")

    def test_simulation(self, model) -> None:
        """Test that the model can be simulated."""
        # Solve the model
        mats = model.system_matrices()
        solution, _info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Simulate
        T_periods = 100
        n_states = model.spec.n_states
        n_shocks = model.spec.n_shocks

        # Initialize
        states = np.zeros((T_periods + 1, n_states))
        shocks = np.random.randn(T_periods, n_shocks) * 0.01  # Small shocks

        # Simulate forward
        for t in range(T_periods):
            states[t + 1, :] = solution.T @ states[t, :] + solution.R @ shocks[t, :]

        # Check that simulation doesn't explode
        assert np.all(np.isfinite(states))
        assert np.max(np.abs(states)) < 100  # Should stay bounded

    def test_irfs(self, model) -> None:
        """Test impulse response functions."""
        # Solve the model
        mats = model.system_matrices()
        solution, _info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Compute IRF to productivity shock (shock index 0)
        shock_idx = 0
        T_irf = 40
        n_states = model.spec.n_states
        n_shocks = model.spec.n_shocks

        irf = np.zeros((T_irf, n_states))
        shock = np.zeros(n_shocks)
        shock[shock_idx] = 1.0  # One std dev shock

        # Initialize with shock
        irf[0, :] = solution.R @ shock

        # Propagate forward
        for t in range(1, T_irf):
            irf[t, :] = solution.T @ irf[t - 1, :]

        # Check that IRFs are finite
        assert np.all(np.isfinite(irf))

        # Check that IRFs are bounded and don't explode
        # Note: SW model has near-unit root variables (productivity), so IRFs may be very persistent
        max_value = np.max(np.abs(irf))
        assert max_value < 100, f"IRF exploding: max value = {max_value}"

        # Verify IRFs eventually stabilize (variance in last 10 periods should be low)
        late_variance = np.var(irf[-10:, :])
        assert late_variance < 1.0, f"IRF not stabilizing: late variance = {late_variance}"

    def test_parameter_priors(self, model) -> None:
        """Test that priors are defined for estimated parameters."""
        estimated_params = [p for p in model.parameters if not p.fixed]

        # Should have multiple estimated parameters
        assert len(estimated_params) > 10

        # Most estimated parameters should have priors
        params_with_priors = [p for p in estimated_params if p.prior is not None]
        assert len(params_with_priors) > 10

    def test_derived_parameters(self, model) -> None:
        """Test computation of derived steady-state parameters."""
        params = model.parameters.to_dict()
        derived = model._compute_steady_state_params(params)

        # Check that key derived parameters exist
        assert "cbeta" in derived
        assert "cgamma" in derived
        assert "cbetabar" in derived
        assert "crk" in derived
        assert "cky" in derived
        assert "ccy" in derived
        assert "ciy" in derived

        # Check reasonable values
        assert 0 < derived["cbeta"] < 1  # Discount factor should be < 1
        assert derived["cgamma"] > 1  # Gross growth rate should be > 1
        assert derived["crk"] > 0  # Rental rate should be positive
        assert 0 < derived["ccy"] < 1  # Consumption share should be between 0 and 1
        assert 0 < derived["ciy"] < 1  # Investment share should be between 0 and 1

    def test_observable_construction(self, model) -> None:
        """Test that observables can be constructed from states."""
        # Solve model
        mats = model.system_matrices()
        solution, _info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Simulate states
        T = 50
        n_states = model.spec.n_states
        n_shocks = model.spec.n_shocks

        states = np.zeros((T, n_states))
        shocks = np.random.randn(T - 1, n_shocks) * 0.01

        for t in range(T - 1):
            states[t + 1, :] = solution.T @ states[t, :] + solution.R @ shocks[t, :]

        # Construct observables
        Z, D = model.measurement_equation()
        observables = states @ Z.T + D

        # Check dimensions
        assert observables.shape == (T, 7)

        # Check that observables are finite
        assert np.all(np.isfinite(observables))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
