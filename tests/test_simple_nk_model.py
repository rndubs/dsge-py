"""Tests for the Simple New Keynesian model."""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.simple_nk_model import create_simple_nk_model
from src.dsge.solvers.linear import solve_linear_model


def test_simple_nk_creation() -> None:
    """Test that the simple NK model can be created."""
    model = create_simple_nk_model()

    assert model.spec.n_states == 9
    assert model.spec.n_controls == 0
    assert model.spec.n_shocks == 3
    assert model.spec.n_observables == 3
    assert len(model.parameters) == 11


def test_simple_nk_system_matrices() -> None:
    """Test that system matrices have correct dimensions."""
    model = create_simple_nk_model()
    mats = model.system_matrices()

    n = model.spec.n_states
    n_shocks = model.spec.n_shocks

    assert mats["Gamma0"].shape == (n, n)
    assert mats["Gamma1"].shape == (n, n)
    assert mats["Psi"].shape == (n, n_shocks)
    assert mats["Pi"].shape == (n, 2)  # 2 expectation errors


def test_simple_nk_measurement() -> None:
    """Test measurement equation."""
    model = create_simple_nk_model()
    Z, D = model.measurement_equation()

    assert Z.shape == (3, 9)  # 3 observables, 9 states
    assert D.shape == (3,)

    # Check that measurement picks up correct states
    assert Z[0, 0] == 1.0  # y
    assert Z[1, 1] == 1.0  # pi
    assert Z[2, 2] == 1.0  # r


def test_simple_nk_solution() -> None:
    """Test that the model can be solved."""
    model = create_simple_nk_model()
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

        # Check that solution is stable (eigenvalues inside unit circle)
        eigvals = np.linalg.eigvals(solution.T)
        max_eigval = np.max(np.abs(eigvals))

        assert max_eigval < 1.0, "Solution should be stable"

    except Exception as e:
        pytest.fail(f"Model solution failed: {e!s}")


def test_simple_nk_simulation() -> None:
    """Test model simulation."""
    model = create_simple_nk_model()
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
    assert np.max(np.abs(states)) < 10.0, "Simulation should not explode"


def test_simple_nk_impulse_responses() -> None:
    """Test impulse response functions."""
    model = create_simple_nk_model()
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
    shock[2] = 1.0  # One std dev monetary policy shock

    # Period 0: initial shock
    irf[0] = solution.R @ shock

    # Periods 1 onwards: propagation
    for h in range(1, H):
        irf[h] = solution.T @ irf[h - 1]

    # Check IRF properties
    # Monetary policy shock should:
    # - Initially increase interest rate
    # - Decrease output (negative)
    # - Decrease inflation (negative)

    assert irf[0, 2] > 0, "MP shock should increase interest rate on impact"
    assert np.min(irf[:, 0]) < 0, "MP shock should decrease output"
    assert np.min(irf[:, 1]) < 0, "MP shock should decrease inflation"

    # IRF should die out
    assert np.max(np.abs(irf[-5:, :])) < np.max(np.abs(irf[:5, :])), "IRF should decay over time"


if __name__ == "__main__":
    # Run tests
    test_simple_nk_creation()

    test_simple_nk_system_matrices()

    test_simple_nk_measurement()

    test_simple_nk_solution()

    test_simple_nk_simulation()

    test_simple_nk_impulse_responses()
