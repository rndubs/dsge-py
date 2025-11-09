"""Tests for linear solvers."""

import numpy as np
import pytest
from dsge.solvers.linear import solve_linear_model, simulate


def test_solve_ar1():
    """Test solving a simple AR(1) model."""
    # AR(1): x_t = ρ*x_{t-1} + ε_t with ρ = 0.9
    Gamma0 = np.array([[1.0]])
    Gamma1 = np.array([[0.9]])
    Psi = np.array([[1.0]])
    Pi = np.array([[1e-10]])

    solution, info = solve_linear_model(Gamma0, Gamma1, Psi, Pi, n_states=1)

    assert info['is_stable']
    assert np.allclose(solution.T, [[0.9]], atol=1e-6)
    assert np.allclose(solution.R, [[1.0]], atol=1e-6)


def test_solve_var2():
    """Test solving a VAR(1) with 2 variables."""
    # x1_t = 0.5*x1_{t-1} + 0.1*x2_{t-1} + ε1_t
    # x2_t = 0.2*x1_{t-1} + 0.7*x2_{t-1} + ε2_t
    Gamma0 = np.eye(2)
    Gamma1 = np.array([[0.5, 0.1],
                       [0.2, 0.7]])
    Psi = np.eye(2)
    Pi = np.eye(2) * 1e-10

    solution, info = solve_linear_model(Gamma0, Gamma1, Psi, Pi, n_states=2)

    assert info['is_stable']
    assert np.allclose(solution.T, Gamma1, atol=1e-6)


def test_unstable_model():
    """Test that unstable models are detected."""
    # x_t = 1.5*x_{t-1} + ε_t (explosive)
    Gamma0 = np.array([[1.0]])
    Gamma1 = np.array([[1.5]])
    Psi = np.array([[1.0]])
    Pi = np.array([[1e-10]])

    solution, info = solve_linear_model(Gamma0, Gamma1, Psi, Pi, n_states=1)

    assert not info['is_stable']


def test_simulate():
    """Test simulation."""
    # Simple AR(1)
    from dsge.solvers.linear import LinearSolution

    solution = LinearSolution(
        T=np.array([[0.9]]),
        R=np.array([[1.0]]),
        C=np.zeros(1),
        Z=np.eye(1),
        D=np.zeros(1),
        Q=np.eye(1),
        n_unstable=0,
        n_states=1,
        is_stable=True
    )

    states, obs = simulate(solution, n_periods=100, random_seed=42)

    assert states.shape == (100, 1)
    assert obs.shape == (100, 1)
    assert np.isfinite(states).all()
    assert np.isfinite(obs).all()
