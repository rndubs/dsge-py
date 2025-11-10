"""
OccBin model estimation using Sequential Monte Carlo.

This module provides estimation capabilities for DSGE models with
occasionally binding constraints.
"""

from dataclasses import dataclass

import numpy as np

from ..models.base import DSGEModel
from ..solvers.occbin import OccBinConstraint
from .likelihood import log_likelihood_occbin
from .smc import SMCResults, SMCSampler


@dataclass
class OccBinSMCResults(SMCResults):
    """
    Results from OccBin SMC estimation.

    Extends SMCResults with OccBin-specific diagnostics.

    Attributes:
    ----------
    particles : array (n_particles x n_params)
        Final particle positions
    weights : array (n_particles,)
        Final particle weights
    log_evidence : float
        Log marginal likelihood estimate
    log_likelihoods : array (n_particles,)
        Log likelihoods for each particle
    acceptance_rate : float
        Overall acceptance rate
    n_iterations : int
        Number of tempering stages
    regime_diagnostics : dict, optional
        OccBin-specific diagnostics (regime counts, etc.)
    """

    regime_diagnostics: dict | None = None


class OccBinSMCSampler(SMCSampler):
    """
    Sequential Monte Carlo sampler for OccBin DSGE models.

    This extends the standard SMC sampler to handle models with
    occasionally binding constraints.
    """

    def __init__(
        self,
        model_M1: DSGEModel,
        model_M2: DSGEModel,
        constraint: OccBinConstraint,
        n_particles: int = 1000,
        n_phi: int = 100,
        lambda_param: float = 2.0,
        resample_threshold: float = 0.5,
        max_filter_iter: int = 50,
    ) -> None:
        """
        Initialize OccBin SMC sampler.

        Parameters
        ----------
        model_M1 : DSGEModel
            Reference regime model (constraint slack)
        model_M2 : DSGEModel
            Alternative regime model (constraint binding)
        constraint : OccBinConstraint
            Constraint specification
        n_particles : int
            Number of particles
        n_phi : int
            Maximum number of tempering stages
        lambda_param : float
            Target for effective sample size ratio
        resample_threshold : float
            Threshold for resampling (ESS / N)
        max_filter_iter : int
            Maximum iterations for OccBin filter
        """
        super().__init__(n_particles, n_phi, lambda_param, resample_threshold)
        self.model_M1 = model_M1
        self.model_M2 = model_M2
        self.constraint = constraint
        self.max_filter_iter = max_filter_iter

    def sample(
        self, data: np.ndarray, n_mh_steps: int = 1, verbose: bool = True
    ) -> OccBinSMCResults:
        """
        Run SMC sampler for OccBin model.

        Parameters
        ----------
        data : array (T x n_obs)
            Observed data
        n_mh_steps : int
            Number of Metropolis-Hastings steps per stage
        verbose : bool
            Print progress information

        Returns:
        -------
        OccBinSMCResults
            Estimation results with OccBin diagnostics
        """
        # Get free parameters (use model_M1, both models share parameters)
        free_params = self.model_M1.parameters.get_free_params()
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
        if verbose:
            pass
        log_likelihoods = np.array(
            [
                self._log_likelihood_tempered(data, particles[i], 0.0)
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
                data, particles, log_likelihoods, phi_new, n_mh_steps
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
                    self._log_likelihood_tempered(data, particles[i], phi)
                    for i in range(self.n_particles)
                ]
            )

        overall_acceptance_rate = (
            n_accepted_total / n_proposed_total if n_proposed_total > 0 else 0.0
        )

        # Compute OccBin diagnostics (optional)
        regime_diagnostics = self._compute_regime_diagnostics(data, particles, weights)

        return OccBinSMCResults(
            particles=particles,
            weights=weights,
            log_evidence=log_evidence,
            log_likelihoods=log_likelihoods,
            acceptance_rate=overall_acceptance_rate,
            n_iterations=iteration,
            regime_diagnostics=regime_diagnostics,
        )

    def _log_likelihood_tempered(self, data: np.ndarray, params: np.ndarray, phi: float) -> float:
        """Tempered log posterior: log prior + phi * log likelihood."""
        self.model_M1.parameters.set_values(params)
        self.model_M2.parameters.set_values(params)
        log_prior = self.model_M1.parameters.log_prior()

        if not np.isfinite(log_prior):
            return -np.inf

        # Evaluate OccBin likelihood
        log_lik = log_likelihood_occbin(
            self.model_M1,
            self.model_M2,
            self.constraint,
            data,
            params,
            max_iter=self.max_filter_iter,
        )

        return log_prior + phi * (log_lik - log_prior)

    def _mutation_step(
        self,
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
        proposal_cov = scale * cov + 1e-8 * np.eye(n_params)

        for _ in range(n_steps):
            for i in range(n_particles):
                # Propose new particle
                proposal = np.random.multivariate_normal(particles[i], proposal_cov)

                # Evaluate posterior
                log_post_proposal = self._log_likelihood_tempered(data, proposal, phi)
                log_post_current = log_posts[i]

                # Accept/reject
                log_alpha = log_post_proposal - log_post_current
                if np.log(np.random.uniform()) < log_alpha:
                    particles[i] = proposal
                    log_posts[i] = log_post_proposal
                    n_accepted += 1

        return particles, n_accepted

    def _compute_regime_diagnostics(
        self, data: np.ndarray, particles: np.ndarray, weights: np.ndarray
    ) -> dict:
        """
        Compute OccBin-specific diagnostics.

        This evaluates regime sequences for the weighted particles.
        """
        # For now, just return basic info
        # Could be extended to track regime sequences, binding frequencies, etc.
        return {"n_particles": len(particles), "effective_sample_size": 1.0 / np.sum(weights**2)}


def estimate_occbin(
    model_M1: DSGEModel,
    model_M2: DSGEModel,
    constraint: OccBinConstraint,
    data: np.ndarray,
    n_particles: int = 1000,
    n_mh_steps: int = 1,
    max_filter_iter: int = 50,
    verbose: bool = True,
) -> OccBinSMCResults:
    """
    Estimate OccBin DSGE model using SMC.

    Parameters
    ----------
    model_M1 : DSGEModel
        Reference regime model (constraint slack)
    model_M2 : DSGEModel
        Alternative regime model (constraint binding)
    constraint : OccBinConstraint
        Constraint specification
    data : array (T x n_obs)
        Observed data
    n_particles : int
        Number of particles
    n_mh_steps : int
        MH steps per tempering stage
    max_filter_iter : int
        Maximum OccBin filter iterations
    verbose : bool
        Print progress

    Returns:
    -------
    OccBinSMCResults
        Estimation results with OccBin diagnostics
    """
    sampler = OccBinSMCSampler(
        model_M1=model_M1,
        model_M2=model_M2,
        constraint=constraint,
        n_particles=n_particles,
        max_filter_iter=max_filter_iter,
    )
    return sampler.sample(data, n_mh_steps=n_mh_steps, verbose=verbose)
