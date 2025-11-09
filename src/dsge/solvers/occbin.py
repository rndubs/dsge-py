"""
OccBin solver for occasionally binding constraints.

Implements the Guerrieri-Iacoviello (2015) algorithm for solving DSGE models
with occasionally binding constraints.
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
from dataclasses import dataclass
from scipy import linalg
from .linear import solve_linear_model, LinearSolution


@dataclass
class OccBinConstraint:
    """
    Specification of an occasionally binding constraint.

    Attributes
    ----------
    name : str
        Name of the constraint (e.g., 'ZLB')
    binding_condition : callable
        Function that takes state vector and returns True if constraint binds
    variable_index : int
        Index of the variable subject to constraint
    bound_value : float
        The bound value (e.g., 0 for ZLB)
    """
    name: str
    binding_condition: Callable[[np.ndarray], bool]
    variable_index: int
    bound_value: float

    def is_binding(self, X: np.ndarray) -> bool:
        """Check if constraint is binding given state X."""
        return self.binding_condition(X)

    def can_relax(self, X: np.ndarray, epsilon: float = 1e-6) -> bool:
        """Check if binding constraint can be relaxed."""
        # For a lower bound constraint, can relax if variable > bound + epsilon
        return X[self.variable_index] > self.bound_value + epsilon


@dataclass
class OccBinSolution:
    """
    Solution to an OccBin model.

    Attributes
    ----------
    states : array
        Time path of state variables
    controls : array
        Time path of control variables
    regime_sequence : array
        Which regime was active each period (0=reference, 1=alternative)
    converged : bool
        Whether the solution converged
    n_iterations : int
        Number of iterations required
    """
    states: np.ndarray
    controls: np.ndarray
    regime_sequence: np.ndarray
    converged: bool
    n_iterations: int


class OccBinSolver:
    """
    OccBin solver for models with occasionally binding constraints.

    This implements the Guerrieri-Iacoviello (2015) guess-and-verify algorithm.
    """

    def __init__(self,
                 solution_M1: LinearSolution,
                 solution_M2: LinearSolution,
                 constraint: OccBinConstraint,
                 max_iter: int = 50,
                 convergence_tol: float = 1e-10):
        """
        Initialize OccBin solver.

        Parameters
        ----------
        solution_M1 : LinearSolution
            Reference regime solution (constraint slack)
        solution_M2 : LinearSolution
            Alternative regime solution (constraint binding)
        constraint : OccBinConstraint
            Constraint specification
        max_iter : int
            Maximum iterations for convergence
        convergence_tol : float
            Tolerance for convergence check
        """
        self.M1 = solution_M1
        self.M2 = solution_M2
        self.constraint = constraint
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol

    def solve(self,
              initial_state: np.ndarray,
              shocks: np.ndarray,
              T: int) -> OccBinSolution:
        """
        Solve the OccBin model.

        Parameters
        ----------
        initial_state : array
            Initial state vector
        shocks : array (T x n_shocks)
            Sequence of structural shocks
        T : int
            Simulation horizon

        Returns
        -------
        OccBinSolution
            Solution with regime-switching paths
        """
        n_states = self.M1.n_states

        # Initialize regime sequence (start with reference regime)
        regime_seq = np.zeros(T, dtype=int)

        converged = False
        iteration = 0

        while not converged and iteration < self.max_iter:
            iteration += 1

            # Generate solution given current regime sequence
            states, controls = self._solve_with_regime_sequence(
                initial_state, shocks, regime_seq, T
            )

            # Check constraints and update regime sequence
            new_regime_seq = self._update_regime_sequence(states, controls, regime_seq)

            # Check convergence
            if np.array_equal(new_regime_seq, regime_seq):
                converged = True

            regime_seq = new_regime_seq

        return OccBinSolution(
            states=states,
            controls=controls,
            regime_sequence=regime_seq,
            converged=converged,
            n_iterations=iteration
        )

    def _solve_with_regime_sequence(self,
                                    initial_state: np.ndarray,
                                    shocks: np.ndarray,
                                    regime_seq: np.ndarray,
                                    T: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve model forward given regime sequence.

        Uses the appropriate decision rules (M1 or M2) for each period.
        """
        n_states = self.M1.n_states
        n_shocks = shocks.shape[1] if len(shocks.shape) > 1 else 1

        states = np.zeros((T, n_states))
        states[0] = initial_state

        # For now, assume no separate control variables (pure state model)
        # This will be extended later
        controls = np.zeros((T, 0))

        for t in range(1, T):
            if regime_seq[t-1] == 0:
                # Use reference regime M1
                states[t] = (self.M1.T @ states[t-1] +
                           self.M1.R @ shocks[t-1] +
                           self.M1.C)
            else:
                # Use alternative regime M2
                states[t] = (self.M2.T @ states[t-1] +
                           self.M2.R @ shocks[t-1] +
                           self.M2.C)

        return states, controls

    def _update_regime_sequence(self,
                               states: np.ndarray,
                               controls: np.ndarray,
                               old_regime_seq: np.ndarray) -> np.ndarray:
        """
        Update regime sequence based on constraint violations.

        Check if:
        - Constraint is violated in periods assumed slack (switch to M2)
        - Constraint can be relaxed in periods assumed binding (switch to M1)
        """
        T = states.shape[0]
        new_regime_seq = old_regime_seq.copy()

        for t in range(T):
            # Combine states and controls for constraint evaluation
            X_t = states[t]  # For now, just use states

            if old_regime_seq[t] == 0:
                # Reference regime - check if constraint violated
                if self.constraint.is_binding(X_t):
                    new_regime_seq[t] = 1  # Switch to alternative
            else:
                # Alternative regime - check if constraint can be relaxed
                if self.constraint.can_relax(X_t):
                    new_regime_seq[t] = 0  # Switch to reference

        return new_regime_seq


def create_zlb_constraint(variable_index: int,
                         bound: float = 0.0,
                         name: str = 'ZLB') -> OccBinConstraint:
    """
    Create a Zero Lower Bound (ZLB) constraint.

    Parameters
    ----------
    variable_index : int
        Index of the interest rate variable
    bound : float
        The lower bound (typically 0)
    name : str
        Name of the constraint

    Returns
    -------
    OccBinConstraint
        ZLB constraint specification
    """
    def binding_condition(X: np.ndarray) -> bool:
        # Binding if variable <= bound (with small tolerance)
        return X[variable_index] <= bound + 1e-10

    return OccBinConstraint(
        name=name,
        binding_condition=binding_condition,
        variable_index=variable_index,
        bound_value=bound
    )
