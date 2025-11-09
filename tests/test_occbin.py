"""Tests for OccBin solver."""

import numpy as np
import pytest
from dsge.solvers.linear import LinearSolution
from dsge.solvers.occbin import (OccBinSolver, OccBinConstraint,
                                 create_zlb_constraint)


def test_zlb_constraint_creation():
    """Test ZLB constraint creation."""
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    assert constraint.name == 'ZLB'
    assert constraint.variable_index == 0
    assert constraint.bound_value == 0.0


def test_constraint_binding():
    """Test constraint binding detection."""
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Should bind when variable <= 0
    X_negative = np.array([-0.5, 1.0, 2.0])
    assert constraint.is_binding(X_negative)

    X_zero = np.array([0.0, 1.0, 2.0])
    assert constraint.is_binding(X_zero)

    # Should not bind when variable > 0
    X_positive = np.array([0.5, 1.0, 2.0])
    assert not constraint.is_binding(X_positive)


def test_constraint_relaxation():
    """Test constraint relaxation detection."""
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Can relax when variable > bound + epsilon
    X_positive = np.array([0.1, 1.0, 2.0])
    assert constraint.can_relax(X_positive)

    # Cannot relax when variable <= bound
    X_negative = np.array([-0.1, 1.0, 2.0])
    assert not constraint.can_relax(X_negative)


def test_occbin_solver_simple():
    """Test OccBin solver with simple example."""
    # Create simple AR(1) models for two regimes
    # M1: x_t = 0.9*x_{t-1} + ε_t
    # M2: x_t = 0.5*x_{t-1} + ε_t

    solution_M1 = LinearSolution(
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

    solution_M2 = LinearSolution(
        T=np.array([[0.5]]),
        R=np.array([[1.0]]),
        C=np.zeros(1),
        Z=np.eye(1),
        D=np.zeros(1),
        Q=np.eye(1),
        n_unstable=0,
        n_states=1,
        is_stable=True
    )

    # Create constraint: switch to M2 when x < -1
    def binding_condition(X):
        return X[0] < -1.0

    constraint = OccBinConstraint(
        name='test',
        binding_condition=binding_condition,
        variable_index=0,
        bound_value=-1.0
    )

    # Create solver
    solver = OccBinSolver(solution_M1, solution_M2, constraint, max_iter=20)

    # Solve with initial negative shock
    T = 20
    initial_state = np.zeros(1)
    shocks = np.zeros((T, 1))
    shocks[0, 0] = -2.0  # Large negative shock

    result = solver.solve(initial_state, shocks, T)

    # Should converge
    assert result.converged
    assert result.n_iterations > 0
    assert result.states.shape == (T, 1)
    assert result.regime_sequence.shape == (T,)


def test_occbin_solver_no_binding():
    """Test OccBin solver when constraint never binds."""
    # Same models as before
    solution_M1 = LinearSolution(
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

    solution_M2 = LinearSolution(
        T=np.array([[0.5]]),
        R=np.array([[1.0]]),
        C=np.zeros(1),
        Z=np.eye(1),
        D=np.zeros(1),
        Q=np.eye(1),
        n_unstable=0,
        n_states=1,
        is_stable=True
    )

    constraint = create_zlb_constraint(variable_index=0, bound=-10.0)  # Very low bound

    solver = OccBinSolver(solution_M1, solution_M2, constraint, max_iter=20)

    # Small shock - should never hit constraint
    T = 20
    initial_state = np.zeros(1)
    shocks = np.zeros((T, 1))
    shocks[0, 0] = -0.5  # Small shock

    result = solver.solve(initial_state, shocks, T)

    # Should converge immediately with all periods in M1
    assert result.converged
    assert result.n_iterations <= 2  # Should converge in 1-2 iterations
    assert np.all(result.regime_sequence == 0)  # All periods in reference regime
