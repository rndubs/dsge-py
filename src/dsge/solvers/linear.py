"""
Linear solution methods for DSGE models.

This module implements solution methods for linear(ized) DSGE models,
including the Blanchard-Kahn (generalized Schur decomposition) method.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg


@dataclass
class LinearSolution:
    """
    Container for linear DSGE model solution.

    The solution is of the form:
        s_t = T * s_{t-1} + R * ε_t + C
        y_t = Z * s_t + D

    where s_t are states, ε_t are shocks, and y_t are observables.
    """

    T: np.ndarray  # Transition matrix (n_states x n_states)
    R: np.ndarray  # Shock loading matrix (n_states x n_shocks)
    C: np.ndarray  # Constant term (n_states,)
    Z: np.ndarray  # Measurement matrix (n_obs x n_states)
    D: np.ndarray  # Measurement constant (n_obs,)
    Q: np.ndarray  # Shock covariance (n_shocks x n_shocks)
    n_unstable: int  # Number of unstable eigenvalues
    n_states: int  # Number of state variables
    is_stable: bool  # Whether solution is stable


def solve_linear_model(
    Gamma0: np.ndarray,
    Gamma1: np.ndarray,
    Psi: np.ndarray,
    Pi: np.ndarray,
    n_states: int,
    tol: float = 1e-6,
) -> tuple[LinearSolution, dict[str, Any]]:
    """
    Solve a linear DSGE model using the Blanchard-Kahn method.

    The model is of the form:
        Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

    where s_t = [states_t; controls_t] and η_t are expectational errors.

    Parameters
    ----------
    Gamma0 : array (n_total x n_total)
        Matrix multiplying current period variables
    Gamma1 : array (n_total x n_total)
        Matrix multiplying lagged variables
    Psi : array (n_total x n_shocks)
        Matrix multiplying shocks
    Pi : array (n_total x n_expect)
        Matrix multiplying expectational errors
    n_states : int
        Number of state variables (vs. controls)
    tol : float
        Tolerance for stability (eigenvalues with |λ| < 1 + tol are stable)

    Returns:
    -------
    solution : LinearSolution
        The model solution
    info : dict
        Diagnostic information including eigenvalues and stability
    """
    n_total = Gamma0.shape[0]
    n_controls = n_total - n_states
    n_shocks = Psi.shape[1]

    # Special case: pure state model (no controls)
    if n_controls == 0:
        # Direct solution: x_t = Γ0^{-1} * Γ1 * x_{t-1} + Γ0^{-1} * Ψ * ε_t
        try:
            Gamma0_inv = linalg.inv(Gamma0)
            T = Gamma0_inv @ Gamma1
            R = Gamma0_inv @ Psi

            # Check stability: all eigenvalues of T should have |λ| < 1
            eigenvalues = linalg.eigvals(T)
            is_stable = bool(np.all(np.abs(eigenvalues) < 1 + tol))

            # Real part for numerical stability
            T = np.real(T)
            R = np.real(R)

            solution = LinearSolution(
                T=T,
                R=R,
                C=np.zeros(n_states),
                Z=np.eye(n_states),
                D=np.zeros(n_states),
                Q=np.eye(n_shocks),
                n_unstable=0 if is_stable else n_states,
                n_states=n_states,
                is_stable=is_stable,
            )

            info = {
                "eigenvalues": eigenvalues,
                "n_stable": n_states if is_stable else 0,
                "n_unstable": 0 if is_stable else n_states,
                "condition": "stable (pure state model)"
                if is_stable
                else "unstable (pure state model)",
                "is_determinate": is_stable,
                "is_stable": is_stable,
            }

            return solution, info
        except linalg.LinAlgError:
            # Singular Gamma0
            msg = "Gamma0 is singular - cannot solve system"
            raise ValueError(msg)

    # Perform generalized Schur decomposition for models with controls
    # This solves the generalized eigenvalue problem: Γ0 * s_t = Γ1 * s_{t-1}
    AA, BB, Q, Z = linalg.qz(Gamma0, Gamma1, output="complex")

    # Compute generalized eigenvalues
    eigenvalues = np.array(
        [AA[i, i] / BB[i, i] if abs(BB[i, i]) > 1e-10 else np.inf for i in range(n_total)]
    )

    # Count stable eigenvalues (|λ| < 1)
    stable_mask = np.abs(eigenvalues) < 1 + tol
    n_stable = np.sum(stable_mask)
    n_unstable = n_total - n_stable

    # Check Blanchard-Kahn condition
    # Number of unstable eigenvalues should equal number of control variables
    is_determinate = n_unstable == n_controls
    is_stable = is_determinate

    if not is_determinate:
        if n_unstable < n_controls:
            condition = "indeterminate (multiple equilibria)"
        else:
            condition = "explosive (no stable equilibrium)"
    else:
        condition = "determinate (unique stable equilibrium)"

    # Reorder Schur decomposition to put stable eigenvalues first
    # This is the ordered QZ decomposition
    AA, BB, _alpha, _beta, Q, Z = linalg.ordqz(AA, BB, sort="iuc")  # inside unit circle

    # Extract blocks
    # After reordering: stable eigenvalues are in top-left block
    Z11 = Z[:n_states, :n_states]
    Z12 = Z[:n_states, n_states:]
    Z[n_states:, :n_states]
    Z[n_states:, n_states:]

    # Solution for state transition
    # s_{1,t} = Z11^{-1} * Z12 * s_{2,t}
    # where s_1 are predetermined states and s_2 are non-predetermined
    if is_determinate and np.abs(linalg.det(Z11)) > 1e-10:
        # For proper solution, we need the policy function
        AA[:n_states, :n_states]
        AA[:n_states, n_states:]
        BB[:n_states, :n_states]
        BB[:n_states, n_states:]

        # Transition matrix for states
        T = linalg.solve(Z11, Z12)

        # Shock loading matrix
        # Transform Psi and Pi using Q
        Psi_tilde = Q.T @ Psi
        Pi_tilde = Q.T @ Pi

        # Extract state block
        Psi_1 = Psi_tilde[:n_states, :]
        Pi_tilde[:n_states, :]

        # Compute shock loading
        if np.abs(linalg.det(Z11)) > 1e-10:
            R = linalg.solve(Z11, Psi_1)
        else:
            R = np.zeros((n_states, n_shocks))

        # Real part for numerical stability (eigenvalues come in conjugate pairs)
        T = np.real(T)
        R = np.real(R)

    else:
        # No stable solution
        T = np.full((n_states, n_states), np.nan)
        R = np.full((n_states, n_shocks), np.nan)

    # Create solution object (with placeholder measurement equation)
    C = np.zeros(n_states)
    Z_meas = np.eye(n_states)  # Identity by default
    D = np.zeros(n_states)
    Q_cov = np.eye(n_shocks)  # Unit variance shocks by default

    solution = LinearSolution(
        T=T,
        R=R,
        C=C,
        Z=Z_meas,
        D=D,
        Q=Q_cov,
        n_unstable=n_unstable,
        n_states=n_states,
        is_stable=is_stable,
    )

    info = {
        "eigenvalues": eigenvalues,
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "condition": condition,
        "is_determinate": is_determinate,
        "is_stable": is_stable,
    }

    return solution, info


def simulate(
    solution: LinearSolution,
    n_periods: int,
    shocks: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the solved DSGE model.

    Parameters
    ----------
    solution : LinearSolution
        Solved model
    n_periods : int
        Number of periods to simulate
    shocks : array (n_periods x n_shocks), optional
        Shock realizations. If None, drawn from N(0, Q)
    initial_state : array (n_states,), optional
        Initial state. If None, starts at zero.
    random_seed : int, optional
        Random seed for shock generation

    Returns:
    -------
    states : array (n_periods x n_states)
        Simulated state variables
    observables : array (n_periods x n_obs)
        Simulated observables
    """
    if not solution.is_stable:
        msg = "Cannot simulate unstable solution"
        raise ValueError(msg)

    n_states = solution.T.shape[0]
    n_shocks = solution.R.shape[1]
    n_obs = solution.Z.shape[0]

    # Initialize
    states = np.zeros((n_periods, n_states))
    observables = np.zeros((n_periods, n_obs))

    if initial_state is not None:
        states[0] = initial_state

    # Generate shocks if not provided
    if shocks is None:
        if random_seed is not None:
            np.random.seed(random_seed)
        # Draw from multivariate normal with covariance Q
        shocks = np.random.multivariate_normal(np.zeros(n_shocks), solution.Q, size=n_periods)

    # Simulate
    for t in range(n_periods):
        if t > 0:
            states[t] = solution.T @ states[t - 1] + solution.R @ shocks[t] + solution.C
        else:
            states[t] = states[t] + solution.R @ shocks[t] + solution.C

        observables[t] = solution.Z @ states[t] + solution.D

    return states, observables
