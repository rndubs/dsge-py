"""Tests for the Galí (2010) unemployment model."""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gali_2010_unemployment import create_gali_2010_model
from src.dsge.solvers.linear import solve_linear_model


def test_gali_2010_creation() -> None:
    """Test that the Galí (2010) model can be created."""
    model = create_gali_2010_model()

    assert model.spec.n_states == 29
    assert model.spec.n_controls == 0
    assert model.spec.n_shocks == 2
    assert model.spec.n_observables == 6
    assert len(model.parameters) == 26


def test_gali_2010_parameter_calibration() -> None:
    """Test that parameters match Galí (2010) calibration."""
    model = create_gali_2010_model()
    params = model.parameters.to_dict()

    # Check key calibration values from p. 515-516
    assert params["N"] == 0.59, "Employment rate should be 0.59"
    assert params["U"] == 0.03, "Unemployment rate should be 0.03"
    assert params["x"] == 0.7, "Job finding rate should be 0.7"
    assert params["alfa"] == 1.0/3.0, "Capital share should be 1/3"
    assert params["betta"] == 0.99, "Discount factor should be 0.99"
    assert params["varphi"] == 5.0, "Frisch elasticity should be 5"
    assert params["theta_w"] == 0.75, "Wage stickiness should be 0.75"
    assert params["theta_p"] == 0.75, "Price stickiness should be 0.75"
    assert params["phi_pi"] == 1.5, "Taylor rule inflation coeff should be 1.5"
    assert params["phi_y"] == 0.125, "Taylor rule output coeff should be 0.125"
    assert params["rho_a"] == 0.9, "Technology shock persistence should be 0.9"
    assert params["rho_nu"] == 0.5, "Monetary shock persistence should be 0.5"


def test_gali_2010_system_matrices() -> None:
    """Test that system matrices have correct dimensions."""
    model = create_gali_2010_model()
    mats = model.system_matrices()

    n = model.spec.n_states
    n_shocks = model.spec.n_shocks

    assert mats["Gamma0"].shape == (n, n)
    assert mats["Gamma1"].shape == (n, n)
    assert mats["Psi"].shape == (n, n_shocks)
    assert mats["Pi"].shape == (n, 6)  # 6 expectation errors

    # Check that matrices are not all zeros
    assert np.any(mats["Gamma0"] != 0), "Gamma0 should have non-zero elements"
    assert np.any(mats["Gamma1"] != 0), "Gamma1 should have non-zero elements"
    assert np.any(mats["Psi"] != 0), "Psi should have non-zero elements"
    assert np.any(mats["Pi"] != 0), "Pi should have non-zero elements"


def test_gali_2010_measurement() -> None:
    """Test measurement equation."""
    model = create_gali_2010_model()
    Z, D = model.measurement_equation()

    assert Z.shape == (6, 29)  # 6 observables, 29 states
    assert D.shape == (6,)

    # Check that measurement matrix is not all zeros
    assert np.any(Z != 0), "Measurement matrix should have non-zero elements"


def test_gali_2010_steady_state() -> None:
    """Test steady state computation."""
    model = create_gali_2010_model()
    ss = model.steady_state()

    assert ss.shape == (29,)
    # For log-linearized model, steady state should be zero
    assert np.allclose(ss, 0.0), "Steady state should be zero for log-linearized model"


def test_gali_2010_solution() -> None:
    """Test that the model can be solved."""
    model = create_gali_2010_model()
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

        # Check eigenvalues
        eigvals = np.linalg.eigvals(solution.T)
        max_eigval = np.max(np.abs(eigvals))

        print(f"Max eigenvalue magnitude: {max_eigval:.4f}")
        assert max_eigval < 10.0, "Solution eigenvalues should be bounded"

    except Exception as e:
        pytest.fail(f"Model solution failed: {e!s}")


def test_gali_2010_simulation() -> None:
    """Test model simulation."""
    model = create_gali_2010_model()
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
    shocks = np.random.randn(T, n_shocks) * 0.01  # Small shocks

    # Simulate forward
    for t in range(1, T):
        states[t] = solution.C + solution.T @ states[t - 1] + solution.R @ shocks[t]

    # Check that simulation doesn't explode
    assert np.all(np.isfinite(states)), "Simulation should remain finite"
    assert np.max(np.abs(states)) < 100.0, "Simulation should not explode"


def test_gali_2010_impulse_responses() -> None:
    """Test impulse response functions."""
    model = create_gali_2010_model()
    mats = model.system_matrices()

    # Solve the model
    solution, _info = solve_linear_model(
        Gamma0=mats["Gamma0"],
        Gamma1=mats["Gamma1"],
        Psi=mats["Psi"],
        Pi=mats["Pi"],
        n_states=model.spec.n_states,
    )

    # Compute IRF to monetary policy shock
    H = 20  # horizon
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    irf = np.zeros((H, n_states))
    shock = np.zeros(n_shocks)
    shock[1] = 1.0  # One std dev monetary policy shock

    # Period 0: initial shock
    irf[0] = solution.R @ shock

    # Periods 1 onwards: propagation
    for h in range(1, H):
        irf[h] = solution.T @ irf[h - 1]

    # Check IRF properties
    assert np.all(np.isfinite(irf)), "IRFs should remain finite"
    assert np.max(np.abs(irf[:5, :])) > 0, "Monetary shock should have non-zero impact"


def test_gali_2010_labor_market_variables() -> None:
    """Test that labor market variables are properly included."""
    model = create_gali_2010_model()

    # Check that labor market variables are in state names
    state_names = model.spec.state_names
    assert "nhat" in state_names, "Employment should be a state variable"
    assert "uhat" in state_names, "Unemployment should be a state variable"
    assert "urhat" in state_names, "Unemployment rate should be a state variable"
    assert "xhat" in state_names, "Job finding rate should be a state variable"
    assert "hhat" in state_names, "New hiring should be a state variable"
    assert "ghat" in state_names, "Hiring costs should be a state variable"
    assert "fhat" in state_names, "Labor force should be a state variable"


def test_gali_2010_equilibrium_conditions() -> None:
    """Test that all 21 equilibrium conditions are implemented."""
    model = create_gali_2010_model()
    mats = model.system_matrices()

    # Count non-zero rows in Gamma0 (each equilibrium condition)
    non_zero_rows = np.sum(np.any(mats["Gamma0"] != 0, axis=1))

    # Should have 21 equilibrium conditions + 8 lag definitions = 29 total
    assert non_zero_rows == 29, f"Expected 29 equations, got {non_zero_rows}"


def test_gali_2010_published_reference() -> None:
    """Test that model has proper documentation."""
    model = create_gali_2010_model()

    # Check that model class has a docstring
    docstring = model.__class__.__doc__
    assert docstring is not None, "Model should have a docstring"
    assert len(docstring) > 100, "Model should have substantial documentation"

    # Check for key terms
    assert "2010" in docstring or "unemployment" in docstring.lower(), "Should reference the model"


if __name__ == "__main__":
    # Run tests
    print("Running Galí (2010) model tests...\n")

    print("Test 1: Model creation")
    test_gali_2010_creation()
    print("✓ Model creation test passed\n")

    print("Test 2: Parameter calibration")
    test_gali_2010_parameter_calibration()
    print("✓ Parameter calibration test passed\n")

    print("Test 3: System matrices")
    test_gali_2010_system_matrices()
    print("✓ System matrices test passed\n")

    print("Test 4: Measurement equation")
    test_gali_2010_measurement()
    print("✓ Measurement equation test passed\n")

    print("Test 5: Steady state")
    test_gali_2010_steady_state()
    print("✓ Steady state test passed\n")

    print("Test 6: Model solution")
    test_gali_2010_solution()
    print("✓ Model solution test passed\n")

    print("Test 7: Simulation")
    test_gali_2010_simulation()
    print("✓ Simulation test passed\n")

    print("Test 8: Impulse responses")
    test_gali_2010_impulse_responses()
    print("✓ Impulse responses test passed\n")

    print("Test 9: Labor market variables")
    test_gali_2010_labor_market_variables()
    print("✓ Labor market variables test passed\n")

    print("Test 10: Equilibrium conditions")
    test_gali_2010_equilibrium_conditions()
    print("✓ Equilibrium conditions test passed\n")

    print("Test 11: Published reference")
    test_gali_2010_published_reference()
    print("✓ Published reference test passed\n")

    print("All tests passed! ✓")
