"""Tests for OccBin filtering."""

import numpy as np

from dsge.filters.occbin_filter import OccBinParticleFilter, occbin_filter
from dsge.solvers.linear import LinearSolution
from dsge.solvers.occbin import create_zlb_constraint


def create_test_solutions():
    """Create simple test solutions for two regimes."""
    # M1: Normal regime (x_t = 0.9*x_{t-1} + eps_t)
    solution_M1 = LinearSolution(
        T=np.array([[0.9]]),
        R=np.array([[1.0]]),
        C=np.zeros(1),
        Z=np.eye(1),
        D=np.zeros(1),
        Q=np.array([[0.1**2]]),
        n_unstable=0,
        n_states=1,
        is_stable=True,
    )

    # M2: Constrained regime (x_t = 0.5*x_{t-1} + eps_t)
    solution_M2 = LinearSolution(
        T=np.array([[0.5]]),
        R=np.array([[1.0]]),
        C=np.zeros(1),
        Z=np.eye(1),
        D=np.zeros(1),
        Q=np.array([[0.1**2]]),
        n_unstable=0,
        n_states=1,
        is_stable=True,
    )

    return solution_M1, solution_M2


def test_occbin_filter_basic() -> None:
    """Test basic OccBin filtering."""
    # Create solutions
    solution_M1, solution_M2 = create_test_solutions()

    # Create constraint (switch when x < -1)
    constraint = create_zlb_constraint(variable_index=0, bound=-1.0)

    # Generate some data
    np.random.seed(42)
    T = 30
    obs = np.random.randn(T, 1) * 0.1

    # Add a few periods with large negative values to trigger constraint
    obs[10:15] = -1.5 + np.random.randn(5, 1) * 0.1

    # Measurement matrices
    Z = np.eye(1)
    D = np.zeros(1)
    H = np.eye(1) * 0.01

    # Run filter
    results = occbin_filter(
        y=obs,
        solution_M1=solution_M1,
        solution_M2=solution_M2,
        constraint=constraint,
        Z=Z,
        D=D,
        H=H,
        max_iter=20,
    )

    # Check basic properties
    assert results.filtered_states.shape == (T, 1)
    assert results.regime_sequence.shape == (T,)
    assert results.n_iterations > 0
    assert np.isfinite(results.log_likelihood)

    # Should have some periods in alternative regime around the big negative shock
    assert np.sum(results.regime_sequence == 1) > 0


def test_occbin_filter_convergence() -> None:
    """Test that OccBin filter converges."""
    solution_M1, solution_M2 = create_test_solutions()
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Simple data
    np.random.seed(42)
    T = 20
    obs = np.random.randn(T, 1) * 0.1

    Z = np.eye(1)
    D = np.zeros(1)
    H = np.eye(1) * 0.01

    results = occbin_filter(
        y=obs,
        solution_M1=solution_M1,
        solution_M2=solution_M2,
        constraint=constraint,
        Z=Z,
        D=D,
        H=H,
        max_iter=10,
    )

    # Should converge in reasonable number of iterations
    assert results.n_iterations < 10


def test_occbin_filter_no_regime_switch() -> None:
    """Test OccBin filter when constraint never binds."""
    solution_M1, solution_M2 = create_test_solutions()

    # Very low bound - should never hit
    constraint = create_zlb_constraint(variable_index=0, bound=-10.0)

    # Small positive shocks
    np.random.seed(42)
    T = 20
    obs = np.abs(np.random.randn(T, 1) * 0.1)  # All positive

    Z = np.eye(1)
    D = np.zeros(1)
    H = np.eye(1) * 0.01

    results = occbin_filter(
        y=obs,
        solution_M1=solution_M1,
        solution_M2=solution_M2,
        constraint=constraint,
        Z=Z,
        D=D,
        H=H,
        max_iter=10,
    )

    # All periods should be in reference regime
    assert np.all(results.regime_sequence == 0)
    assert results.n_iterations <= 2  # Should converge immediately


def test_occbin_particle_filter_basic() -> None:
    """Test basic particle filter for OccBin."""
    solution_M1, solution_M2 = create_test_solutions()
    constraint = create_zlb_constraint(variable_index=0, bound=-0.5)

    # Generate data with regime switch
    np.random.seed(42)
    T = 20
    obs = np.random.randn(T, 1) * 0.2
    obs[8:12] = -1.0  # Force constraint binding

    Z = np.eye(1)
    D = np.zeros(1)
    H = np.eye(1) * 0.01

    # Create and run particle filter
    pf = OccBinParticleFilter(
        solution_M1=solution_M1,
        solution_M2=solution_M2,
        constraint=constraint,
        n_particles=100,  # Small for speed
    )

    results = pf.filter(y=obs, Z=Z, D=D, H=H)

    # Check results
    assert results.filtered_states.shape == (T, 1)
    assert results.regime_sequence.shape == (T,)
    assert results.regime_probabilities is not None
    assert results.regime_probabilities.shape == (T, 2)
    assert np.isfinite(results.log_likelihood)

    # Regime probabilities should sum to 1
    assert np.allclose(np.sum(results.regime_probabilities, axis=1), 1.0)


def test_occbin_filter_missing_data() -> None:
    """Test OccBin filter with missing observations."""
    solution_M1, solution_M2 = create_test_solutions()
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Data with missing observations
    np.random.seed(42)
    T = 20
    obs = np.random.randn(T, 1) * 0.1
    obs[5:8] = np.nan  # Missing data

    Z = np.eye(1)
    D = np.zeros(1)
    H = np.eye(1) * 0.01

    results = occbin_filter(
        y=obs,
        solution_M1=solution_M1,
        solution_M2=solution_M2,
        constraint=constraint,
        Z=Z,
        D=D,
        H=H,
        max_iter=10,
    )

    # Should handle missing data gracefully
    assert np.isfinite(results.log_likelihood)
    assert results.filtered_states.shape == (T, 1)
    # Filtered states should be finite even for missing obs periods
    assert np.isfinite(results.filtered_states).all()
