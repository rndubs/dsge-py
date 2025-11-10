"""Base classes for DSGE model specification."""

from abc import ABC, abstractmethod

import numpy as np

from .parameters import ParameterSet


class ModelSpecification:
    """Specification of a DSGE model's structure."""

    def __init__(
        self,
        n_states: int,
        n_controls: int,
        n_shocks: int,
        n_observables: int,
        state_names: list[str] | None = None,
        control_names: list[str] | None = None,
        shock_names: list[str] | None = None,
        observable_names: list[str] | None = None,
    ) -> None:
        """
        Initialize model specification.

        Parameters
        ----------
        n_states : int
            Number of state variables
        n_controls : int
            Number of control variables
        n_shocks : int
            Number of shocks
        n_observables : int
            Number of observable variables
        state_names : list of str, optional
            Names of state variables
        control_names : list of str, optional
            Names of control variables
        shock_names : list of str, optional
            Names of shocks
        observable_names : list of str, optional
            Names of observable variables
        """
        self.n_states = n_states
        self.n_controls = n_controls
        self.n_shocks = n_shocks
        self.n_observables = n_observables

        self.state_names = state_names or [f"s{i}" for i in range(n_states)]
        self.control_names = control_names or [f"c{i}" for i in range(n_controls)]
        self.shock_names = shock_names or [f"e{i}" for i in range(n_shocks)]
        self.observable_names = observable_names or [f"y{i}" for i in range(n_observables)]

        # Validate dimensions
        assert len(self.state_names) == n_states
        assert len(self.control_names) == n_controls
        assert len(self.shock_names) == n_shocks
        assert len(self.observable_names) == n_observables


class DSGEModel(ABC):
    """
    Abstract base class for DSGE models.

    This class defines the interface that all DSGE models must implement.
    """

    def __init__(self, spec: ModelSpecification) -> None:
        """
        Initialize DSGE model.

        Parameters
        ----------
        spec : ModelSpecification
            Model specification defining dimensions and variable names
        """
        self.spec = spec
        self.parameters = ParameterSet()
        self._setup_parameters()

    @abstractmethod
    def _setup_parameters(self):
        """
        Define model parameters.

        This method should be implemented by subclasses to add all
        model parameters to self.parameters.
        """

    @abstractmethod
    def system_matrices(self, params: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Compute the linearized system matrices.

        For a linear DSGE model of the form:
            Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

        where s_t includes both states and controls, ε_t are shocks,
        and η_t are expectational errors.

        Parameters
        ----------
        params : array, optional
            Parameter values. If None, use current parameter values.

        Returns:
        -------
        dict
            Dictionary containing 'Gamma0', 'Gamma1', 'Psi', 'Pi' matrices
        """

    @abstractmethod
    def measurement_equation(
        self, params: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Define the measurement equation linking states to observables.

        y_t = Z * s_t + D + η_t

        Parameters
        ----------
        params : array, optional
            Parameter values. If None, use current parameter values.

        Returns:
        -------
        Z : array
            Measurement matrix (n_obs x n_states)
        D : array
            Constant term (n_obs,)
        """

    def shock_covariance(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Covariance matrix of structural shocks.

        Parameters
        ----------
        params : array, optional
            Parameter values. If None, use current parameter values.

        Returns:
        -------
        Q : array
            Shock covariance matrix (n_shocks x n_shocks)
        """
        # Default: diagonal matrix with unit variances
        return np.eye(self.spec.n_shocks)

    def measurement_error_covariance(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Covariance matrix of measurement errors.

        Parameters
        ----------
        params : array, optional
            Parameter values. If None, use current parameter values.

        Returns:
        -------
        R : array
            Measurement error covariance matrix (n_obs x n_obs)
        """
        # Default: no measurement error
        return np.zeros((self.spec.n_observables, self.spec.n_observables))

    def steady_state(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the steady state of the model.

        Parameters
        ----------
        params : array, optional
            Parameter values. If None, use current parameter values.

        Returns:
        -------
        ss : array
            Steady state values (n_states + n_controls,)
        """
        # Default: zero steady state (model in deviations from steady state)
        return np.zeros(self.spec.n_states + self.spec.n_controls)

    def validate(self) -> bool:
        """
        Validate model specification.

        Returns:
        -------
        valid : bool
            True if model is valid
        """
        # Check that system matrices have correct dimensions
        try:
            mats = self.system_matrices()
            n_total = self.spec.n_states + self.spec.n_controls

            assert mats["Gamma0"].shape == (n_total, n_total)
            assert mats["Gamma1"].shape == (n_total, n_total)
            assert mats["Psi"].shape == (n_total, self.spec.n_shocks)
            assert mats["Pi"].shape[0] == n_total

            Z, D = self.measurement_equation()
            assert Z.shape == (self.spec.n_observables, n_total)
            assert D.shape == (self.spec.n_observables,)

            return True
        except Exception:
            return False
