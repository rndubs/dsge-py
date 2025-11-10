"""
Sequential Monte Carlo (SMC) sampler for Bayesian estimation of DSGE models.

This implements a basic SMC sampler for posterior inference.
"""

from dataclasses import dataclass

import numpy as np

from ..models.base import DSGEModel
from .likelihood import log_likelihood_linear


@dataclass
class SMCResults:
    """Results from SMC estimation."""

    particles: np.ndarray  # Final particle positions (n_particles x n_params)
    weights: np.ndarray  # Final particle weights (n_particles,)
    log_evidence: float  # Log marginal likelihood estimate
    log_likelihoods: np.ndarray  # Log likelihoods for each particle
    acceptance_rate: float  # Overall acceptance rate
    n_iterations: int  # Number of tempering stages


class SMCSampler:
    """
    Sequential Monte Carlo sampler for DSGE model estimation.

    This implements adaptive tempering SMC as described in
    Herbst & Schorfheide (2014).
    """

    def __init__(
        self,
        n_particles: int = 1000,
        n_phi: int = 100,
        lambda_param: float = 2.0,
        resample_threshold: float = 0.5,
    ) -> None:
        """
        Initialize SMC sampler.

        Parameters
        ----------
        n_particles : int
            Number of particles
        n_phi : int
            Maximum number of tempering stages
        lambda_param : float
            Target for effective sample size ratio
        resample_threshold : float
            Threshold for resampling (ESS / N)
        """
        self.n_particles = n_particles
        self.n_phi = n_phi
        self.lambda_param = lambda_param
        self.resample_threshold = resample_threshold

    def sample(
        self, model: DSGEModel, data: np.ndarray, n_mh_steps: int = 1, verbose: bool = True
    ) -> SMCResults:
        """
        Run SMC sampler.

        Parameters
        ----------
        model : DSGEModel
            DSGE model to estimate
        data : array (T x n_obs)
            Observed data
        n_mh_steps : int
            Number of Metropolis-Hastings steps per stage
        verbose : bool
            Print progress information

        Returns:
        -------
        SMCResults
            Estimation results
        """
        # Get free parameters
        free_params = model.parameters.get_free_params()
        param_names = list(free_params.keys())
        n_params = len(param_names)

        if n_params == 0:
            msg = "No free parameters to estimate"
            raise ValueError(msg)

        # Initialize particles from prior
        particles = np.zeros((self.n_particles, n_params))
        for i, (name, param) in enumerate(free_params.items()):
            if param.prior is None:
                msg = f"Parameter {name} has no prior distribution"
                raise ValueError(msg)
            particles[:, i] = param.prior.rvs(self.n_particles)

        # Initialize weights (uniform)
        weights = np.ones(self.n_particles) / self.n_particles

        # Evaluate initial log likelihoods
        log_likelihoods = np.array(
            [
                self._log_likelihood_tempered(model, data, particles[i], 0.0)
                for i in range(self.n_particles)
            ]
        )

        # Initialize tempering schedule
        phi = 0.0  # Start at prior (phi=0), move to posterior (phi=1)
        phi_sequence = [0.0]
        log_evidence = 0.0
        n_accepted_total = 0
        n_proposed_total = 0

        iteration = 0
        while phi < 1.0 and iteration < self.n_phi:
            iteration += 1

            # Adapt tempering parameter
            phi_new = self._adapt_tempering(phi, log_likelihoods, weights)
            phi_increment = phi_new - phi

            if verbose:
                pass

            # Reweight particles
            incremental_weights = np.exp(phi_increment * log_likelihoods)
            weights *= incremental_weights
            weights /= np.sum(weights)

            # Update log evidence
            log_evidence += np.log(np.mean(incremental_weights))

            # Compute effective sample size
            ess = 1.0 / np.sum(weights**2)
            ess_ratio = ess / self.n_particles

            if verbose:
                pass

            # Resample if needed
            if ess_ratio < self.resample_threshold:
                indices = self._systematic_resample(weights)
                particles = particles[indices]
                weights = np.ones(self.n_particles) / self.n_particles
                if verbose:
                    pass

            # Mutation step (Metropolis-Hastings)
            particles, n_accepted = self._mutation_step(
                model, data, particles, log_likelihoods, phi_new, n_mh_steps
            )

            n_accepted_total += n_accepted
            n_proposed_total += self.n_particles * n_mh_steps

            if verbose:
                n_accepted / (self.n_particles * n_mh_steps)

            # Update for next iteration
            phi = phi_new
            phi_sequence.append(phi)

            # Re-evaluate likelihoods at new tempering parameter
            log_likelihoods = np.array(
                [
                    self._log_likelihood_tempered(model, data, particles[i], phi)
                    for i in range(self.n_particles)
                ]
            )

        overall_acceptance_rate = (
            n_accepted_total / n_proposed_total if n_proposed_total > 0 else 0.0
        )

        return SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=log_evidence,
            log_likelihoods=log_likelihoods,
            acceptance_rate=overall_acceptance_rate,
            n_iterations=iteration,
        )

    def _log_likelihood_tempered(
        self, model: DSGEModel, data: np.ndarray, params: np.ndarray, phi: float
    ) -> float:
        """Tempered log posterior: log prior + phi * log likelihood."""
        model.parameters.set_values(params)
        log_prior = model.parameters.log_prior()

        if not np.isfinite(log_prior):
            return -np.inf

        # Evaluate likelihood
        log_lik = log_likelihood_linear(model, data, params)

        return log_prior + phi * (log_lik - log_prior)

    def _adapt_tempering(
        self, phi: float, log_likelihoods: np.ndarray, weights: np.ndarray
    ) -> float:
        """Adapt tempering parameter to achieve target ESS."""
        if phi >= 1.0:
            return 1.0

        # Binary search for phi that gives target ESS
        phi_low = phi
        phi_high = 1.0
        target_ess = self.lambda_param * len(weights)

        for _ in range(20):  # Max iterations
            phi_mid = (phi_low + phi_high) / 2.0

            # Compute ESS at phi_mid
            incremental_weights = np.exp((phi_mid - phi) * log_likelihoods)
            temp_weights = weights * incremental_weights
            temp_weights /= np.sum(temp_weights)
            ess = 1.0 / np.sum(temp_weights**2)

            if abs(ess - target_ess) < 1.0:
                return phi_mid

            if ess < target_ess:
                phi_high = phi_mid
            else:
                phi_low = phi_mid

        return min(phi_high, 1.0)

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling."""
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform(0, 1)) / n
        cumsum = np.cumsum(weights)
        return np.asarray(np.searchsorted(cumsum, positions))

    def _mutation_step(
        self,
        model: DSGEModel,
        data: np.ndarray,
        particles: np.ndarray,
        log_posts: np.ndarray,
        phi: float,
        n_steps: int,
    ) -> tuple[np.ndarray, int]:
        """Metropolis-Hastings mutation step."""
        n_particles, n_params = particles.shape
        n_accepted = 0

        # Compute proposal covariance
        cov = np.cov(particles.T)
        scale = 2.38**2 / n_params  # Optimal scaling
        proposal_cov = scale * cov + 1e-8 * np.eye(n_params)  # Add small diagonal for stability

        for _ in range(n_steps):
            for i in range(n_particles):
                # Propose new particle
                proposal = np.random.multivariate_normal(particles[i], proposal_cov)

                # Evaluate posterior
                log_post_proposal = self._log_likelihood_tempered(model, data, proposal, phi)
                log_post_current = log_posts[i]

                # Accept/reject
                log_alpha = log_post_proposal - log_post_current
                if np.log(np.random.uniform()) < log_alpha:
                    particles[i] = proposal
                    log_posts[i] = log_post_proposal
                    n_accepted += 1

        return particles, n_accepted


def estimate_dsge(
    model: DSGEModel,
    data: np.ndarray,
    n_particles: int = 1000,
    n_mh_steps: int = 1,
    verbose: bool = True,
) -> SMCResults:
    """
    Estimate DSGE model using SMC.

    Parameters
    ----------
    model : DSGEModel
        Model to estimate
    data : array (T x n_obs)
        Observed data
    n_particles : int
        Number of particles
    n_mh_steps : int
        MH steps per tempering stage
    verbose : bool
        Print progress

    Returns:
    -------
    SMCResults
        Estimation results
    """
    sampler = SMCSampler(n_particles=n_particles)
    return sampler.sample(model, data, n_mh_steps=n_mh_steps, verbose=verbose)
