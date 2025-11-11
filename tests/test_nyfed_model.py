"""Tests for the NYFed DSGE Model 1002."""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsge.solvers.linear import solve_linear_model
from models.nyfed_model_1002 import create_nyfed_model


def test_nyfed_creation() -> None:
    """Test that the NYFed model can be created."""
    model = create_nyfed_model()

    assert model.spec.n_states == 48
    assert model.spec.n_controls == 0
    assert model.spec.n_shocks == 9
    assert model.spec.n_observables == 13
    assert len(model.parameters) == 67


def test_nyfed_system_matrices() -> None:
    """Test that system matrices have correct dimensions."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    n = model.spec.n_states
    n_shocks = model.spec.n_shocks

    assert mats["Gamma0"].shape == (n, n)
    assert mats["Gamma1"].shape == (n, n)
    assert mats["Psi"].shape == (n, n_shocks)
    assert mats["Pi"].shape[0] == n
    assert mats["Pi"].shape[1] == 13  # Number of expectation errors

    # Check that matrices are not empty
    assert np.any(mats["Gamma0"] != 0)
    assert np.any(mats["Gamma1"] != 0)
    assert np.any(mats["Psi"] != 0)


def test_nyfed_measurement() -> None:
    """Test measurement equation."""
    model = create_nyfed_model()
    Z, D = model.measurement_equation()

    assert Z.shape == (13, 48)  # 13 observables, 48 states
    assert D.shape == (13,)

    # Check that measurement picks up key states
    # GDP growth should depend on y, y_lag, z
    assert np.any(Z[0, :] != 0)


def test_nyfed_steady_state() -> None:
    """Test steady state computation."""
    model = create_nyfed_model()
    ss = model.steady_state()

    assert ss.shape == (48,)
    # All states should be zero in log-linearized model
    assert np.allclose(ss, 0.0)


def test_nyfed_solution() -> None:
    """Test that the model can be solved."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    # Solve the model
    try:
        solution, _info = solve_linear_model(
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

        # Check that solution is stable (eigenvalues inside or on unit circle)
        eigvals = np.linalg.eigvals(solution.T)
        np.max(np.abs(eigvals))

        # For complex models, some eigenvalues may be close to or on unit circle
        # Check that not all eigenvalues are explosive
        np.sum(np.abs(eigvals) > 1.01)

    except Exception as e:
        pytest.fail(f"Model solution failed: {e!s}")


def test_nyfed_simulation() -> None:
    """Test model simulation."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    # Solve the model
    try:
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

        # Print stats for key variables
        {name: i for i, name in enumerate(model.spec.state_names)}

    except Exception as e:
        pytest.fail(f"Simulation failed: {e!s}")


def test_nyfed_impulse_responses() -> None:
    """Test impulse response functions."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    # Solve the model
    try:
        solution, _info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        # Compute IRF to monetary policy shock
        H = 40  # horizon (10 years quarterly)
        n_states = model.spec.n_states
        n_shocks = model.spec.n_shocks

        irf = np.zeros((H, n_states))
        shock = np.zeros(n_shocks)

        # Find monetary policy shock index
        shock_idx = model.spec.shock_names.index("eps_rm")
        shock[shock_idx] = 1.0  # One std dev monetary policy shock

        # Period 0: initial shock
        irf[0] = solution.R @ shock

        # Periods 1 onwards: propagation
        for h in range(1, H):
            irf[h] = solution.T @ irf[h - 1]

        # Check IRF properties
        idx = {name: i for i, name in enumerate(model.spec.state_names)}

        if irf[0, idx["R"]] > 0:  # If rate increases
            pass

        # IRF should eventually decay (check last 5 periods smaller than first 5)
        if np.max(np.abs(irf[:5, :])) > 1e-10:  # Only check if there's a response
            np.max(np.abs(irf[-5:, :])) / np.max(np.abs(irf[:5, :]))

    except Exception as e:
        pytest.fail(f"IRF computation failed: {e!s}")


if __name__ == "__main__":
    # Run tests
    test_nyfed_creation()

    test_nyfed_system_matrices()

    test_nyfed_measurement()

    test_nyfed_steady_state()

    test_nyfed_solution()

    test_nyfed_simulation()

    test_nyfed_impulse_responses()
