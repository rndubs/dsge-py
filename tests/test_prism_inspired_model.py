"""Tests for the PRISM-inspired model with labor market search frictions."""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.prism_inspired_model import create_prism_inspired_model
from src.dsge.solvers.linear import solve_linear_model


def test_prism_creation() -> None:
    """Test that the PRISM-inspired model can be created."""
    model = create_prism_inspired_model()

    assert model.spec.n_states == 24
    assert model.spec.n_controls == 0
    assert model.spec.n_shocks == 3
    assert model.spec.n_observables == 7
    assert len(model.parameters) == 19


def test_prism_parameter_values() -> None:
    """Test that parameters have sensible default values."""
    model = create_prism_inspired_model()
    params = model.parameters.to_dict()

    # Check key parameters are in reasonable ranges
    assert 0.95 < params["beta"] < 1.0, "Discount factor should be near 1"
    assert params["sigma"] > 0, "IES should be positive"
    assert 0 < params["theta_p"] < 1, "Calvo parameter should be between 0 and 1"
    assert 0 < params["alpha_m"] < 1, "Matching elasticity should be between 0 and 1"
    assert params["rho_u"] > 0, "Separation rate should be positive"
    assert params["phi_pi"] > 1, "Taylor principle should hold"


def test_prism_system_matrices() -> None:
    """Test that system matrices have correct dimensions."""
    model = create_prism_inspired_model()
    mats = model.system_matrices()

    n = model.spec.n_states
    n_shocks = model.spec.n_shocks

    assert mats["Gamma0"].shape == (n, n)
    assert mats["Gamma1"].shape == (n, n)
    assert mats["Psi"].shape == (n, n_shocks)
    assert mats["Pi"].shape == (n, 4)  # 4 expectation errors

    # Check that matrices are not all zeros
    assert np.any(mats["Gamma0"] != 0), "Gamma0 should have non-zero elements"
    assert np.any(mats["Gamma1"] != 0), "Gamma1 should have non-zero elements"
    assert np.any(mats["Psi"] != 0), "Psi should have non-zero elements"
    assert np.any(mats["Pi"] != 0), "Pi should have non-zero elements"


def test_prism_measurement() -> None:
    """Test measurement equation."""
    model = create_prism_inspired_model()
    Z, D = model.measurement_equation()

    assert Z.shape == (7, 24)  # 7 observables, 24 states
    assert D.shape == (7,)

    # Check that measurement matrix is not all zeros
    assert np.any(Z != 0), "Measurement matrix should have non-zero elements"

    # Check that growth rates are properly differenced
    # Output growth should be: dy = y - y_lag
    assert Z[0, 1] == 1.0, "Output growth picks up y"
    assert Z[0, 10] == -1.0, "Output growth differences y_lag"

    # Unemployment rate should directly observe u
    assert Z[6, 6] == 1.0, "Unemployment observable picks up u"


def test_prism_steady_state() -> None:
    """Test steady state computation."""
    model = create_prism_inspired_model()
    ss = model.steady_state()

    assert ss.shape == (24,)
    # For log-linearized model, steady state should be zero
    assert np.allclose(ss, 0.0), "Steady state should be zero for log-linearized model"


def test_prism_solution() -> None:
    """Test that the model can be solved."""
    model = create_prism_inspired_model()
    mats = model.system_matrices()

    # Solve the model
    try:
        solution, info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Check solution exists
        assert solution is not None
        assert solution.T is not None
        assert solution.R is not None
        assert solution.C is not None

        # Check dimensions
        n = model.spec.n_states
        n_shocks = model.spec.n_shocks

        assert solution.T.shape == (n, n)
        assert solution.R.shape == (n, n_shocks)
        assert solution.C.shape == (n,)

        # Check that solution has reasonable eigenvalues
        eigvals = np.linalg.eigvals(solution.T)
        max_eigval = np.max(np.abs(eigvals))

        assert max_eigval < 10.0, "Solution eigenvalues should be bounded"

    except Exception as e:
        pytest.fail(f"Model solution failed: {e!s}")


def test_prism_simulation() -> None:
    """Test model simulation."""
    model = create_prism_inspired_model()
    mats = model.system_matrices()

    # Solve the model
    solution, _info = solve_linear_model(
        Gamma0=mats["Gamma0"],
        Gamma1=mats["Gamma1"],
        Psi=mats["Psi"],
        Pi=mats["Pi"],
        n_states=model.spec.n_states,
    )

    # Simulate
    T = 100
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    # Initialize
    states = np.zeros((T, n_states))
    shocks = np.random.randn(T, n_shocks) * 0.001  # Very small shocks

    # Simulate forward
    for t in range(1, T):
        states[t] = solution.C + solution.T @ states[t - 1] + solution.R @ shocks[t]

    # Check that simulation doesn't explode
    assert np.all(np.isfinite(states)), "Simulation should remain finite"
    # More lenient threshold for this model
    assert np.max(np.abs(states)) < 100.0, "Simulation should not explode"


def test_prism_impulse_responses() -> None:
    """Test impulse response functions."""
    model = create_prism_inspired_model()
    mats = model.system_matrices()

    # Solve the model
    solution, _info = solve_linear_model(
        Gamma0=mats["Gamma0"],
        Gamma1=mats["Gamma1"],
        Psi=mats["Psi"],
        Pi=mats["Pi"],
        n_states=model.spec.n_states,
    )

    # Compute IRF to technology shock
    H = 40  # horizon
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    irf = np.zeros((H, n_states))
    shock = np.zeros(n_shocks)
    shock[0] = 1.0  # One std dev technology shock

    # Period 0: initial shock
    irf[0] = solution.R @ shock

    # Periods 1 onwards: propagation
    for h in range(1, H):
        irf[h] = solution.T @ irf[h - 1]

    # Check IRF properties
    # All IRFs should remain finite
    assert np.all(np.isfinite(irf)), "IRFs should remain finite"

    # Technology shock should have some persistent effects
    # (exact signs depend on model calibration)
    assert np.max(np.abs(irf[:5, :])) > 0, "Technology shock should have non-zero impact"


def test_prism_labor_market_variables() -> None:
    """Test that labor market variables are properly included."""
    model = create_prism_inspired_model()

    # Check that labor market variables are in state names
    state_names = model.spec.state_names
    assert "n" in state_names, "Employment should be a state variable"
    assert "u" in state_names, "Unemployment should be a state variable"
    assert "theta" in state_names, "Labor market tightness should be a state variable"
    assert "x" in state_names, "Hiring rate should be a state variable"


def test_prism_shock_processes() -> None:
    """Test that shock processes are properly specified."""
    model = create_prism_inspired_model()
    mats = model.system_matrices()
    params = model.parameters.to_dict()

    # Check shock persistence parameters
    assert 0 <= params["rho_a"] < 1, "Technology shock persistence should be in [0,1)"
    assert 0 <= params["rho_m"] < 1, "Monetary shock persistence should be in [0,1)"
    assert 0 <= params["rho_s"] < 1, "Labor market shock persistence should be in [0,1)"

    # Check shock standard deviations
    assert params["sigma_a"] > 0, "Technology shock std dev should be positive"
    assert params["sigma_m"] > 0, "Monetary shock std dev should be positive"
    assert params["sigma_s"] > 0, "Labor market shock std dev should be positive"


def test_prism_observables_consistency() -> None:
    """Test that observables are consistently defined."""
    model = create_prism_inspired_model()
    Z, D = model.measurement_equation()

    # Check that each observable has at least one non-zero entry
    for i in range(model.spec.n_observables):
        assert np.any(Z[i, :] != 0), f"Observable {i} should map to at least one state"

    # Check observable names
    obs_names = model.spec.observable_names
    assert "obs_dy" in obs_names, "Output growth should be observable"
    assert "obs_pi" in obs_names, "Inflation should be observable"
    assert "obs_u" in obs_names, "Unemployment should be observable"


def test_prism_matrix_sparsity() -> None:
    """Test matrix structure and sparsity."""
    model = create_prism_inspired_model()
    mats = model.system_matrices()

    # Calculate sparsity
    total_elements = mats["Gamma0"].size
    nonzero_elements = np.count_nonzero(mats["Gamma0"])
    sparsity = 1 - (nonzero_elements / total_elements)

    # DSGE models typically have sparse matrices
    # This model should have some sparsity
    assert sparsity > 0.1, "System matrices should have some sparsity"

    print(f"Gamma0 sparsity: {sparsity:.2%}")


def test_prism_parameter_priors() -> None:
    """Test that parameter priors are accessible (if defined)."""
    model = create_prism_inspired_model()

    # Check that parameters can be retrieved
    for param in model.parameters:
        assert hasattr(param, "name"), "Parameter should have a name"
        assert hasattr(param, "value"), "Parameter should have a value"
        assert hasattr(param, "fixed"), "Parameter should have fixed attribute"
        assert hasattr(param, "description"), "Parameter should have a description"


if __name__ == "__main__":
    # Run tests
    print("Running PRISM-inspired model tests...\n")

    print("Test 1: Model creation")
    test_prism_creation()
    print("✓ Model creation test passed\n")

    print("Test 2: Parameter values")
    test_prism_parameter_values()
    print("✓ Parameter values test passed\n")

    print("Test 3: System matrices")
    test_prism_system_matrices()
    print("✓ System matrices test passed\n")

    print("Test 4: Measurement equation")
    test_prism_measurement()
    print("✓ Measurement equation test passed\n")

    print("Test 5: Steady state")
    test_prism_steady_state()
    print("✓ Steady state test passed\n")

    print("Test 6: Model solution")
    test_prism_solution()
    print("✓ Model solution test passed\n")

    print("Test 7: Simulation")
    test_prism_simulation()
    print("✓ Simulation test passed\n")

    print("Test 8: Impulse responses")
    test_prism_impulse_responses()
    print("✓ Impulse responses test passed\n")

    print("Test 9: Labor market variables")
    test_prism_labor_market_variables()
    print("✓ Labor market variables test passed\n")

    print("Test 10: Shock processes")
    test_prism_shock_processes()
    print("✓ Shock processes test passed\n")

    print("Test 11: Observables consistency")
    test_prism_observables_consistency()
    print("✓ Observables consistency test passed\n")

    print("Test 12: Matrix sparsity")
    test_prism_matrix_sparsity()
    print("✓ Matrix sparsity test passed\n")

    print("Test 13: Parameter priors")
    test_prism_parameter_priors()
    print("✓ Parameter priors test passed\n")

    print("All tests passed! ✓")
