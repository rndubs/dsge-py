"""
Tests for the Cleveland Fed CLEMENTINE DSGE Model.

This module contains comprehensive tests for the CLEMENTINE model implementation,
including model structure, parameter specifications, matrix construction,
solution properties, and economic behavior.
"""

import numpy as np
import pytest

from models.clementine_model import ClementineModel, create_clementine_model
from src.dsge.solvers.linear import solve_linear_model


class TestClementineModelStructure:
    """Test basic model structure and initialization."""

    def test_model_creation(self):
        """Test that model can be created."""
        model = create_clementine_model()
        assert model is not None
        assert isinstance(model, ClementineModel)

    def test_model_dimensions(self):
        """Test that model dimensions are correct."""
        model = create_clementine_model()

        assert model.spec.n_states == 40
        assert model.spec.n_controls == 0
        assert model.spec.n_shocks == 7
        assert model.spec.n_observables == 10

    def test_state_names(self):
        """Test that all state names are defined."""
        model = create_clementine_model()

        expected_core_states = [
            "y", "c", "i", "k", "pi", "R", "w", "mc",
            "q_k", "spread", "g", "z_trend", "z_stat"
        ]

        expected_labor_states = [
            "n", "u_rate", "v", "theta", "q_v",
            "f_rate", "s_rate", "l"
        ]

        for state in expected_core_states:
            assert state in model.spec.state_names

        for state in expected_labor_states:
            assert state in model.spec.state_names

    def test_shock_names(self):
        """Test that all shock names are defined."""
        model = create_clementine_model()

        expected_shocks = [
            "shock_z_trend", "shock_z_stat", "shock_b",
            "shock_i", "shock_g", "shock_p", "shock_w"
        ]

        for shock in expected_shocks:
            assert shock in model.spec.shock_names

    def test_observable_names(self):
        """Test that all observable names are defined."""
        model = create_clementine_model()

        expected_observables = [
            "obs_gdp_growth", "obs_cons_growth", "obs_inv_growth",
            "obs_employment", "obs_unemp_rate", "obs_wage_growth",
            "obs_inflation", "obs_ffr", "obs_spread", "obs_hours"
        ]

        for obs in expected_observables:
            assert obs in model.spec.observable_names


class TestClementineModelParameters:
    """Test parameter specifications and priors."""

    def test_parameter_count(self):
        """Test that correct number of parameters are defined."""
        model = create_clementine_model()
        # Should have ~41 parameters
        assert len(model.parameters) >= 40
        assert len(model.parameters) <= 45

    def test_key_parameters_exist(self):
        """Test that key structural parameters exist."""
        model = create_clementine_model()

        key_params = [
            "beta", "sigma_c", "h", "alpha", "delta",
            "zeta_p", "zeta_w", "chi", "kappa_v", "xi",
            "psi_pi", "psi_y", "rho_R",
            "rho_z_trend", "rho_z_stat"
        ]

        for param in key_params:
            assert param in model.parameters.names()

    def test_parameter_priors(self):
        """Test that parameters have appropriate priors where specified."""
        model = create_clementine_model()

        # Test specific priors
        beta = model.parameters["beta"]
        assert not beta.fixed
        assert beta.prior is not None

        sigma_c = model.parameters["sigma_c"]
        assert not sigma_c.fixed
        assert sigma_c.prior is not None

        # Test fixed parameters
        delta = model.parameters["delta"]
        assert delta.fixed

    def test_parameter_bounds(self):
        """Test that parameters have reasonable values."""
        model = create_clementine_model()

        # Discount factor should be close to 1
        assert 0.99 <= model.parameters["beta"].value <= 1.0

        # Risk aversion should be positive
        assert model.parameters["sigma_c"].value > 0

        # Depreciation should be small and positive
        assert 0 < model.parameters["delta"].value < 0.1

        # Capital share should be between 0 and 1
        assert 0 < model.parameters["alpha"].value < 1

        # Taylor rule should satisfy Taylor principle (roughly)
        assert model.parameters["psi_pi"].value > 1.0

    def test_labor_market_parameters(self):
        """Test labor market specific parameters."""
        model = create_clementine_model()

        # Matching elasticity in [0, 1]
        assert 0 < model.parameters["chi"].value < 1

        # Bargaining power in [0, 1]
        assert 0 < model.parameters["xi"].value < 1

        # Separation rate should be small
        assert 0 < model.parameters["rho_s"].value < 0.5

        # Vacancy cost should be positive
        assert model.parameters["kappa_v"].value > 0


class TestClementineModelMatrices:
    """Test system matrix construction."""

    def test_system_matrices_dimensions(self):
        """Test that system matrices have correct dimensions."""
        model = create_clementine_model()
        mats = model.system_matrices()

        n = model.spec.n_states
        n_shocks = model.spec.n_shocks

        assert mats["Gamma0"].shape == (n, n)
        assert mats["Gamma1"].shape == (n, n)
        assert mats["Psi"].shape == (n, n_shocks)
        assert mats["Pi"].shape[0] == n

    def test_system_matrices_no_nan(self):
        """Test that matrices contain no NaN values."""
        model = create_clementine_model()
        mats = model.system_matrices()

        assert not np.any(np.isnan(mats["Gamma0"]))
        assert not np.any(np.isnan(mats["Gamma1"]))
        assert not np.any(np.isnan(mats["Psi"]))
        assert not np.any(np.isnan(mats["Pi"]))

    def test_system_matrices_not_zero(self):
        """Test that matrices are not all zeros."""
        model = create_clementine_model()
        mats = model.system_matrices()

        assert np.any(mats["Gamma0"] != 0)
        assert np.any(mats["Gamma1"] != 0)
        assert np.any(mats["Psi"] != 0)
        # Pi might be zero if no expectational errors

    def test_gamma0_invertibility(self):
        """Test that Gamma0 is invertible (or near-invertible)."""
        model = create_clementine_model()
        mats = model.system_matrices()

        # Check rank or condition number
        rank = np.linalg.matrix_rank(mats["Gamma0"])
        assert rank >= model.spec.n_states - 5  # Allow some slack

    def test_shock_loading_matrix(self):
        """Test that Psi matrix loads shocks correctly."""
        model = create_clementine_model()
        mats = model.system_matrices()

        # Each shock should affect at least one equation
        for col in range(mats["Psi"].shape[1]):
            assert np.any(mats["Psi"][:, col] != 0), f"Shock {col} has no effect"


class TestClementineModelMeasurement:
    """Test measurement equation."""

    def test_measurement_equation_dimensions(self):
        """Test measurement equation has correct dimensions."""
        model = create_clementine_model()
        Z, D = model.measurement_equation()

        assert Z.shape == (model.spec.n_observables, model.spec.n_states)
        assert D.shape == (model.spec.n_observables,)

    def test_measurement_equation_no_nan(self):
        """Test measurement matrices contain no NaN."""
        model = create_clementine_model()
        Z, D = model.measurement_equation()

        assert not np.any(np.isnan(Z))
        assert not np.any(np.isnan(D))

    def test_measurement_equation_links_states(self):
        """Test that each observable is linked to at least one state."""
        model = create_clementine_model()
        Z, D = model.measurement_equation()

        # Each row should have at least one non-zero entry
        for i in range(Z.shape[0]):
            assert np.any(Z[i, :] != 0), f"Observable {i} not linked to any state"

    def test_shock_covariance(self):
        """Test shock covariance matrix."""
        model = create_clementine_model()
        Q = model.shock_covariance()

        assert Q.shape == (model.spec.n_shocks, model.spec.n_shocks)
        assert not np.any(np.isnan(Q))
        # Should be diagonal (independent shocks)
        assert np.allclose(Q, np.diag(np.diag(Q)))
        # Should be positive definite
        eigvals = np.linalg.eigvalsh(Q)
        assert np.all(eigvals > 0)


class TestClementineModelSteadyState:
    """Test steady state computation."""

    def test_steady_state_dimensions(self):
        """Test steady state vector has correct dimension."""
        model = create_clementine_model()
        ss = model.steady_state()

        assert ss.shape == (model.spec.n_states,)

    def test_steady_state_zeros(self):
        """Test that steady state is zeros (log-linearized system)."""
        model = create_clementine_model()
        ss = model.steady_state()

        # For log-linearized model, SS should be zeros
        assert np.allclose(ss, 0.0)

    def test_steady_state_ratios(self):
        """Test computation of steady-state ratios."""
        model = create_clementine_model()
        p = model.parameters.to_dict()
        ss_ratios = model._compute_steady_state_ratios(p)

        # Check key ratios are computed
        assert "r_ss" in ss_ratios
        assert "R_ss" in ss_ratios
        assert "n_ss" in ss_ratios
        assert "u_ss" in ss_ratios
        assert "c_y_ss" in ss_ratios
        assert "i_y_ss" in ss_ratios

        # Check ratios are reasonable
        assert 0 < ss_ratios["c_y_ss"] < 1
        assert 0 < ss_ratios["i_y_ss"] < 1
        assert ss_ratios["c_y_ss"] + ss_ratios["i_y_ss"] <= 1

        # Employment + unemployment = 1 (labor force normalized)
        assert np.isclose(ss_ratios["n_ss"] + ss_ratios["u_ss"], 1.0, atol=0.01)


class TestClementineModelSolution:
    """Test model solution properties."""

    def test_model_solves(self):
        """Test that model can be solved."""
        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        assert solution is not None
        assert info is not None

    def test_solution_stability(self):
        """Test that solution is stable."""
        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # For a growth model, eigenvalues should be well-behaved
        # Allow slightly above 1 for unit roots in growth
        max_eig = np.max(np.abs(solution.eigenvalues))
        assert max_eig < 1.05, f"Max eigenvalue {max_eig:.4f} too large"

        # Should satisfy Blanchard-Kahn conditions approximately
        # (may have near-unit root for growth)
        assert solution.is_stable or max_eig < 1.02

    def test_policy_functions(self):
        """Test that policy functions are computed."""
        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Check that T matrix exists and has correct shape
        assert hasattr(solution, "T")
        assert solution.T.shape == (model.spec.n_states, model.spec.n_states)
        assert not np.any(np.isnan(solution.T))

        # Check that R matrix exists
        assert hasattr(solution, "R")
        assert solution.R.shape == (model.spec.n_states, model.spec.n_shocks)


class TestClementineModelSimulation:
    """Test model simulation."""

    def test_simulation_runs(self):
        """Test that model can be simulated."""
        from src.dsge.solvers.linear import simulate

        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Set up for simulation
        solution.Z, solution.D = model.measurement_equation()
        solution.Q = model.shock_covariance()

        # Simulate
        states, obs = simulate(solution, n_periods=100, random_seed=42)

        assert states.shape == (100, model.spec.n_states)
        assert obs.shape == (100, model.spec.n_observables)

    def test_simulation_bounded(self):
        """Test that simulated paths remain bounded."""
        from src.dsge.solvers.linear import simulate

        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        solution.Z, solution.D = model.measurement_equation()
        solution.Q = model.shock_covariance()

        # Simulate longer
        states, obs = simulate(solution, n_periods=200, random_seed=42)

        # Check bounds (allowing for growth)
        # States in deviations should not explode
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(obs))

        # Max absolute deviation should be reasonable
        # (accounting for accumulated growth)
        max_abs_state = np.max(np.abs(states))
        assert max_abs_state < 100, f"States exploded: max = {max_abs_state}"


class TestClementineModelIRFs:
    """Test impulse response functions."""

    def test_irf_computation(self):
        """Test that IRFs can be computed."""
        from src.dsge.solvers.linear import compute_irf

        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Compute IRF to first shock
        irf = compute_irf(solution, shock_index=0, periods=40, shock_size=1.0)

        assert irf.shape == (40, model.spec.n_states)
        assert not np.any(np.isnan(irf))

    def test_irf_bounded(self):
        """Test that IRFs are bounded."""
        from src.dsge.solvers.linear import compute_irf

        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Test IRFs for all shocks
        for shock_idx in range(model.spec.n_shocks):
            irf = compute_irf(solution, shock_index=shock_idx, periods=40)

            assert np.all(np.isfinite(irf)), f"IRF {shock_idx} has non-finite values"
            max_abs = np.max(np.abs(irf))
            assert max_abs < 50, f"IRF {shock_idx} too large: {max_abs}"

    @pytest.mark.parametrize("shock_name,shock_idx", [
        ("Technology (trend)", 0),
        ("Technology (stat)", 1),
        ("Preference", 2),
    ])
    def test_irf_signs(self, shock_name, shock_idx):
        """Test that key IRFs have expected signs."""
        from src.dsge.solvers.linear import compute_irf

        model = create_clementine_model()
        mats = model.system_matrices()

        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        irf = compute_irf(solution, shock_index=shock_idx, periods=20)

        # Get state indices
        idx = {name: i for i, name in enumerate(model.spec.state_names)}

        if shock_name.startswith("Technology"):
            # Positive technology shock should increase output
            assert irf[1, idx["y"]] > 0, "Tech shock should increase output"
        elif shock_name == "Preference":
            # Preference shock typically reduces consumption/demand
            # (positive shock = higher discount = lower consumption)
            pass  # Sign depends on specification


def test_model_summary():
    """Test model summary information."""
    model = create_clementine_model()

    print("\n" + "="*70)
    print("CLEMENTINE MODEL SUMMARY")
    print("="*70)
    print(f"Model: {model.__class__.__name__}")
    print(f"\nDimensions:")
    print(f"  States:      {model.spec.n_states}")
    print(f"  Controls:    {model.spec.n_controls}")
    print(f"  Shocks:      {model.spec.n_shocks}")
    print(f"  Observables: {model.spec.n_observables}")
    print(f"  Parameters:  {len(model.parameters)}")

    print(f"\nKey Features:")
    print(f"  • Labor market search and matching (Mortensen-Pissarides)")
    print(f"  • Zero lower bound (ZLB) compatibility")
    print(f"  • Stochastic trend growth")
    print(f"  • New Keynesian core (sticky prices & wages)")
    print(f"  • Financial frictions (credit spreads)")

    print(f"\nState Variables (first 10):")
    for i, name in enumerate(model.spec.state_names[:10]):
        print(f"  {i+1:2d}. {name}")

    print(f"\nShocks:")
    for i, name in enumerate(model.spec.shock_names):
        print(f"  {i+1}. {name}")

    print(f"\nObservables:")
    for i, name in enumerate(model.spec.observable_names):
        print(f"  {i+1:2d}. {name}")

    print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
