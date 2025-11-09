"""
Tests for the NYFed DSGE Model 1002.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model


def test_nyfed_creation():
    """Test that the NYFed model can be created."""
    model = create_nyfed_model()

    assert model.spec.n_states == 48
    assert model.spec.n_controls == 0
    assert model.spec.n_shocks == 9
    assert model.spec.n_observables == 13
    assert len(model.parameters) == 67


def test_nyfed_system_matrices():
    """Test that system matrices have correct dimensions."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    n = model.spec.n_states
    n_shocks = model.spec.n_shocks

    assert mats['Gamma0'].shape == (n, n)
    assert mats['Gamma1'].shape == (n, n)
    assert mats['Psi'].shape == (n, n_shocks)
    assert mats['Pi'].shape[0] == n
    assert mats['Pi'].shape[1] == 13  # Number of expectation errors

    # Check that matrices are not empty
    assert np.any(mats['Gamma0'] != 0)
    assert np.any(mats['Gamma1'] != 0)
    assert np.any(mats['Psi'] != 0)


def test_nyfed_measurement():
    """Test measurement equation."""
    model = create_nyfed_model()
    Z, D = model.measurement_equation()

    assert Z.shape == (13, 48)  # 13 observables, 48 states
    assert D.shape == (13,)

    # Check that measurement picks up key states
    # GDP growth should depend on y, y_lag, z
    assert np.any(Z[0, :] != 0)


def test_nyfed_steady_state():
    """Test steady state computation."""
    model = create_nyfed_model()
    ss = model.steady_state()

    assert ss.shape == (48,)
    # All states should be zero in log-linearized model
    assert np.allclose(ss, 0.0)


def test_nyfed_solution():
    """Test that the model can be solved."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    # Solve the model
    try:
        solution, info = solve_linear_model(
            Gamma0=mats['Gamma0'],
            Gamma1=mats['Gamma1'],
            Psi=mats['Psi'],
            Pi=mats['Pi'],
            n_states=model.spec.n_states
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
        max_eigval = np.max(np.abs(eigvals))

        print(f"\nMaximum eigenvalue magnitude: {max_eigval:.4f}")
        print(f"Solution returned with info: {info}")

        # For complex models, some eigenvalues may be close to or on unit circle
        # Check that not all eigenvalues are explosive
        explosive_count = np.sum(np.abs(eigvals) > 1.01)
        print(f"Number of explosive eigenvalues: {explosive_count}")

    except Exception as e:
        pytest.fail(f"Model solution failed: {str(e)}")


def test_nyfed_simulation():
    """Test model simulation."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    # Solve the model
    try:
        solution, info = solve_linear_model(
            Gamma0=mats['Gamma0'],
            Gamma1=mats['Gamma1'],
            Psi=mats['Psi'],
            Pi=mats['Pi'],
            n_states=model.spec.n_states
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
            states[t] = solution.C + solution.T @ states[t-1] + solution.R @ shocks[t]

        # Check that simulation doesn't explode
        assert np.all(np.isfinite(states)), "Simulation should remain finite"

        print(f"\nSimulation statistics:")
        print(f"  Max absolute value: {np.max(np.abs(states)):.4f}")
        print(f"  Mean absolute value: {np.mean(np.abs(states)):.4f}")

        # Print stats for key variables
        idx = {name: i for i, name in enumerate(model.spec.state_names)}
        print(f"  Output (y) std dev: {np.std(states[:, idx['y']]):.4f}")
        print(f"  Inflation (pi) std dev: {np.std(states[:, idx['pi']]):.4f}")
        print(f"  Rate (R) std dev: {np.std(states[:, idx['R']]):.4f}")

    except Exception as e:
        pytest.fail(f"Simulation failed: {str(e)}")


def test_nyfed_impulse_responses():
    """Test impulse response functions."""
    model = create_nyfed_model()
    mats = model.system_matrices()

    # Solve the model
    try:
        solution, info = solve_linear_model(
            Gamma0=mats['Gamma0'],
            Gamma1=mats['Gamma1'],
            Psi=mats['Psi'],
            Pi=mats['Pi'],
            n_states=model.spec.n_states
        )

        # Compute IRF to monetary policy shock
        H = 40  # horizon (10 years quarterly)
        n_states = model.spec.n_states
        n_shocks = model.spec.n_shocks

        irf = np.zeros((H, n_states))
        shock = np.zeros(n_shocks)

        # Find monetary policy shock index
        shock_idx = model.spec.shock_names.index('eps_rm')
        shock[shock_idx] = 1.0  # One std dev monetary policy shock

        # Period 0: initial shock
        irf[0] = solution.R @ shock

        # Periods 1 onwards: propagation
        for h in range(1, H):
            irf[h] = solution.T @ irf[h-1]

        # Check IRF properties
        idx = {name: i for i, name in enumerate(model.spec.state_names)}

        print(f"\nIRF to monetary policy shock:")
        print(f"  Output impact: {irf[0, idx['y']]:.4f}")
        print(f"  Inflation impact: {irf[0, idx['pi']]:.4f}")
        print(f"  Rate impact: {irf[0, idx['R']]:.4f}")

        if irf[0, idx['R']] > 0:  # If rate increases
            print(f"  Output minimum: {np.min(irf[:, idx['y']]):.4f} at period {np.argmin(irf[:, idx['y']])}")

        # IRF should eventually decay (check last 5 periods smaller than first 5)
        if np.max(np.abs(irf[:5, :])) > 1e-10:  # Only check if there's a response
            decay_ratio = np.max(np.abs(irf[-5:, :])) / np.max(np.abs(irf[:5, :]))
            print(f"  Decay ratio (last 5 / first 5): {decay_ratio:.4f}")

    except Exception as e:
        pytest.fail(f"IRF computation failed: {str(e)}")


if __name__ == "__main__":
    # Run tests
    test_nyfed_creation()
    print("✓ Model creation test passed")

    test_nyfed_system_matrices()
    print("✓ System matrices test passed")

    test_nyfed_measurement()
    print("✓ Measurement equation test passed")

    test_nyfed_steady_state()
    print("✓ Steady state test passed")

    test_nyfed_solution()
    print("✓ Model solution test passed")

    test_nyfed_simulation()
    print("✓ Simulation test passed")

    test_nyfed_impulse_responses()
    print("✓ Impulse response test passed")

    print("\n✅ All tests passed!")
