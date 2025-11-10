"""
OccBin-aware filtering for regime-switching models.

This module extends the Kalman filter to handle models with occasionally
binding constraints, where the state-space representation switches between
regimes.
"""

from dataclasses import dataclass

import numpy as np

from ..solvers.linear import LinearSolution
from ..solvers.occbin import OccBinConstraint


@dataclass
class OccBinFilterResults:
    """
    Results from OccBin filtering.

    Attributes:
    ----------
    log_likelihood : float
        Log likelihood of the data
    filtered_states : array (T x n_states)
        Filtered state estimates
    filtered_covariances : array (T x n_states x n_states)
        Filtered state covariances
    regime_sequence : array (T,)
        Inferred regime sequence (0=reference, 1=alternative)
    regime_probabilities : array (T x 2), optional
        Regime probabilities if using probabilistic filter
    n_iterations : int
        Number of OccBin iterations for convergence
    """

    log_likelihood: float
    filtered_states: np.ndarray
    filtered_covariances: np.ndarray
    regime_sequence: np.ndarray
    regime_probabilities: np.ndarray | None
    n_iterations: int


def occbin_filter(
    y: np.ndarray,
    solution_M1: LinearSolution,
    solution_M2: LinearSolution,
    constraint: OccBinConstraint,
    Z: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    max_iter: int = 50,
) -> OccBinFilterResults:
    """
    Kalman filter for OccBin models with regime-switching.

    This implements a perfect foresight approach where the regime sequence
    is determined iteratively based on constraint violations.

    Parameters
    ----------
    y : array (T x n_obs)
        Observed data
    solution_M1 : LinearSolution
        Reference regime (constraint slack)
    solution_M2 : LinearSolution
        Alternative regime (constraint binding)
    constraint : OccBinConstraint
        Constraint specification
    Z : array (n_obs x n_states)
        Measurement matrix
    D : array (n_obs,)
        Measurement constant
    H : array (n_obs x n_obs)
        Measurement error covariance
    a0 : array (n_states,), optional
        Initial state mean
    P0 : array (n_states x n_states), optional
        Initial state covariance
    max_iter : int
        Maximum OccBin iterations

    Returns:
    -------
    OccBinFilterResults
        Filtering results with regime sequence
    """
    T_periods, _n_obs = y.shape
    n_states = solution_M1.n_states

    # Initialize regime sequence (start with reference regime)
    regime_seq = np.zeros(T_periods, dtype=int)

    # Initialize storage
    filtered_states = np.zeros((T_periods, n_states))
    filtered_covariances = np.zeros((T_periods, n_states, n_states))
    np.zeros((T_periods, n_states))
    np.zeros((T_periods, n_states, n_states))

    # Initial conditions
    if a0 is None:
        a0 = np.zeros(n_states)
    if P0 is None:
        from ..filters.kalman import solve_discrete_lyapunov

        RQR = solution_M1.R @ solution_M1.Q @ solution_M1.R.T
        P0 = solve_discrete_lyapunov(solution_M1.T, RQR)

    converged = False
    iteration = 0
    log_likelihood = -np.inf

    while not converged and iteration < max_iter:
        iteration += 1

        # Run Kalman filter with current regime sequence
        log_lik, filt_states, filt_cov, _pred_states, _pred_cov = _filter_with_regimes(
            y, regime_seq, solution_M1, solution_M2, Z, D, H, a0, P0
        )

        # Check constraints and update regime sequence
        new_regime_seq = _update_regime_sequence_filter(filt_states, constraint, regime_seq)

        # Check convergence
        if np.array_equal(new_regime_seq, regime_seq):
            converged = True
            filtered_states = filt_states
            filtered_covariances = filt_cov
            log_likelihood = log_lik

        regime_seq = new_regime_seq

    return OccBinFilterResults(
        log_likelihood=log_likelihood,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        regime_sequence=regime_seq,
        regime_probabilities=None,  # Not computed in perfect foresight version
        n_iterations=iteration,
    )


def _filter_with_regimes(
    y: np.ndarray,
    regime_seq: np.ndarray,
    solution_M1: LinearSolution,
    solution_M2: LinearSolution,
    Z: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Kalman filter with regime-dependent matrices.

    Uses appropriate transition matrices (T, R, Q) for each regime.
    """
    T_periods, n_obs = y.shape
    n_states = solution_M1.n_states

    # Storage
    filtered_states = np.zeros((T_periods, n_states))
    filtered_covariances = np.zeros((T_periods, n_states, n_states))
    predicted_states = np.zeros((T_periods, n_states))
    predicted_covariances = np.zeros((T_periods, n_states, n_states))

    # Initialize
    a = a0.copy()
    P = P0.copy()
    log_likelihood = 0.0

    for t in range(T_periods):
        # Select regime-specific matrices
        if regime_seq[t] == 0:
            T_t = solution_M1.T
            R_t = solution_M1.R
            Q_t = solution_M1.Q
        else:
            T_t = solution_M2.T
            R_t = solution_M2.R
            Q_t = solution_M2.Q

        # Store predicted state
        predicted_states[t] = a
        predicted_covariances[t] = P

        # Handle missing data
        y_t = y[t]
        if np.any(np.isnan(y_t)):
            # If observation missing, skip update
            filtered_states[t] = a
            filtered_covariances[t] = P
        else:
            # Forecast error
            v = y_t - (Z @ a + D)

            # Forecast error covariance
            F = Z @ P @ Z.T + H

            # Kalman gain
            try:
                F_inv = np.linalg.inv(F)
                K = P @ Z.T @ F_inv

                # Update state estimate
                a_update = a + K @ v
                P_update = P - K @ Z @ P

                filtered_states[t] = a_update
                filtered_covariances[t] = P_update

                # Update log likelihood
                sign, logdet = np.linalg.slogdet(F)
                if sign > 0:
                    log_likelihood += -0.5 * (n_obs * np.log(2 * np.pi) + logdet + v.T @ F_inv @ v)
                else:
                    log_likelihood = -np.inf

                a = a_update
                P = P_update
            except np.linalg.LinAlgError:
                # Singular forecast error covariance
                filtered_states[t] = a
                filtered_covariances[t] = P
                log_likelihood = -np.inf

        # Predict next period
        a = T_t @ a
        P = T_t @ P @ T_t.T + R_t @ Q_t @ R_t.T

        # Ensure symmetry
        P = 0.5 * (P + P.T)

    return (
        log_likelihood,
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
    )


def _update_regime_sequence_filter(
    filtered_states: np.ndarray, constraint: OccBinConstraint, old_regime_seq: np.ndarray
) -> np.ndarray:
    """
    Update regime sequence based on filtered states and constraint.

    Similar to OccBin solver, but uses filtered states instead of
    simulated states.
    """
    T = filtered_states.shape[0]
    new_regime_seq = old_regime_seq.copy()

    for t in range(T):
        X_t = filtered_states[t]

        if old_regime_seq[t] == 0:
            # Reference regime - check if constraint violated
            if constraint.is_binding(X_t):
                new_regime_seq[t] = 1
        # Alternative regime - check if can relax
        elif constraint.can_relax(X_t):
            new_regime_seq[t] = 0

    return new_regime_seq


class OccBinParticleFilter:
    """
    Particle filter for OccBin models.

    This handles regime uncertainty by maintaining particles that track
    both states and regime sequences.
    """

    def __init__(
        self,
        solution_M1: LinearSolution,
        solution_M2: LinearSolution,
        constraint: OccBinConstraint,
        n_particles: int = 1000,
    ) -> None:
        """
        Initialize particle filter.

        Parameters
        ----------
        solution_M1 : LinearSolution
            Reference regime
        solution_M2 : LinearSolution
            Alternative regime
        constraint : OccBinConstraint
            Constraint specification
        n_particles : int
            Number of particles
        """
        self.M1 = solution_M1
        self.M2 = solution_M2
        self.constraint = constraint
        self.n_particles = n_particles

    def filter(
        self, y: np.ndarray, Z: np.ndarray, D: np.ndarray, H: np.ndarray
    ) -> OccBinFilterResults:
        """
        Run particle filter.

        Parameters
        ----------
        y : array (T x n_obs)
            Observed data
        Z : array (n_obs x n_states)
            Measurement matrix
        D : array (n_obs,)
            Measurement constant
        H : array (n_obs x n_obs)
            Measurement error covariance

        Returns:
        -------
        OccBinFilterResults
            Filtering results with regime probabilities
        """
        T_periods, _n_obs = y.shape
        n_states = self.M1.n_states

        # Initialize particles
        particles = np.zeros((self.n_particles, n_states))
        weights = np.ones(self.n_particles) / self.n_particles
        regime_particles = np.zeros((self.n_particles, T_periods), dtype=int)

        # Storage for results
        filtered_states = np.zeros((T_periods, n_states))
        filtered_covariances = np.zeros((T_periods, n_states, n_states))
        regime_probs = np.zeros((T_periods, 2))

        log_likelihood = 0.0

        for t in range(T_periods):
            # Determine regime for each particle
            for i in range(self.n_particles):
                if self.constraint.is_binding(particles[i]):
                    regime_particles[i, t] = 1
                else:
                    regime_particles[i, t] = 0

            # Propagate particles
            new_particles = np.zeros_like(particles)
            for i in range(self.n_particles):
                regime = regime_particles[i, t]
                if regime == 0:
                    T_mat, R_mat, Q_mat = self.M1.T, self.M1.R, self.M1.Q
                else:
                    T_mat, R_mat, Q_mat = self.M2.T, self.M2.R, self.M2.Q

                # Propagate
                shock = np.random.multivariate_normal(np.zeros(self.M1.R.shape[1]), Q_mat)
                new_particles[i] = T_mat @ particles[i] + R_mat @ shock

            # Weight particles based on observation
            if not np.any(np.isnan(y[t])):
                for i in range(self.n_particles):
                    pred_obs = Z @ new_particles[i] + D
                    residual = y[t] - pred_obs
                    weights[i] *= np.exp(-0.5 * residual.T @ np.linalg.inv(H) @ residual)

                # Normalize weights
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights /= weight_sum
                    log_likelihood += np.log(weight_sum)
                else:
                    weights = np.ones(self.n_particles) / self.n_particles

            # Resample if needed (systematic resampling)
            ess = 1.0 / np.sum(weights**2)
            if ess < self.n_particles / 2:
                indices = self._systematic_resample(weights)
                new_particles = new_particles[indices]
                regime_particles = regime_particles[indices]
                weights = np.ones(self.n_particles) / self.n_particles

            particles = new_particles

            # Compute filtered state (weighted average)
            filtered_states[t] = np.average(particles, weights=weights, axis=0)
            filtered_covariances[t] = np.cov(particles.T, aweights=weights)

            # Compute regime probabilities
            regime_probs[t, 0] = np.sum(weights[regime_particles[:, t] == 0])
            regime_probs[t, 1] = np.sum(weights[regime_particles[:, t] == 1])

        # Most likely regime sequence (mode)
        regime_sequence = np.argmax(regime_probs, axis=1)

        return OccBinFilterResults(
            log_likelihood=log_likelihood,
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            regime_sequence=regime_sequence,
            regime_probabilities=regime_probs,
            n_iterations=1,  # Particle filter runs once
        )

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling."""
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform(0, 1)) / n
        cumsum = np.cumsum(weights)
        return np.asarray(np.searchsorted(cumsum, positions))
