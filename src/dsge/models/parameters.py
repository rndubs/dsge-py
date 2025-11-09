"""
Parameter definitions and prior distributions for DSGE models.
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class Prior:
    """Prior distribution for a parameter."""

    distribution: str  # 'normal', 'beta', 'gamma', 'uniform', 'invgamma'
    params: Dict[str, float]  # Distribution parameters

    def __post_init__(self):
        """Validate distribution parameters."""
        valid_dists = ['normal', 'beta', 'gamma', 'uniform', 'invgamma']
        if self.distribution not in valid_dists:
            raise ValueError(f"Distribution must be one of {valid_dists}")

    def logpdf(self, x: float) -> float:
        """Log probability density function."""
        if self.distribution == 'normal':
            return stats.norm.logpdf(x, loc=self.params['mean'], scale=self.params['std'])
        elif self.distribution == 'beta':
            return stats.beta.logpdf(x, a=self.params['alpha'], b=self.params['beta'])
        elif self.distribution == 'gamma':
            return stats.gamma.logpdf(x, a=self.params['shape'], scale=1/self.params['rate'])
        elif self.distribution == 'uniform':
            return stats.uniform.logpdf(x, loc=self.params['lower'],
                                       scale=self.params['upper']-self.params['lower'])
        elif self.distribution == 'invgamma':
            return stats.invgamma.logpdf(x, a=self.params['shape'], scale=self.params['scale'])
        return -np.inf

    def rvs(self, size: int = 1) -> np.ndarray:
        """Random variates from the prior."""
        if self.distribution == 'normal':
            return stats.norm.rvs(loc=self.params['mean'], scale=self.params['std'], size=size)
        elif self.distribution == 'beta':
            return stats.beta.rvs(a=self.params['alpha'], b=self.params['beta'], size=size)
        elif self.distribution == 'gamma':
            return stats.gamma.rvs(a=self.params['shape'], scale=1/self.params['rate'], size=size)
        elif self.distribution == 'uniform':
            return stats.uniform.rvs(loc=self.params['lower'],
                                    scale=self.params['upper']-self.params['lower'], size=size)
        elif self.distribution == 'invgamma':
            return stats.invgamma.rvs(a=self.params['shape'], scale=self.params['scale'], size=size)
        return np.array([])


@dataclass
class Parameter:
    """A single parameter in a DSGE model."""

    name: str
    value: float
    prior: Optional[Prior] = None
    fixed: bool = False
    description: str = ""
    bounds: tuple = field(default_factory=lambda: (-np.inf, np.inf))

    def __post_init__(self):
        """Validate parameter."""
        if self.value < self.bounds[0] or self.value > self.bounds[1]:
            raise ValueError(f"Parameter {self.name} value {self.value} outside bounds {self.bounds}")


class ParameterSet:
    """Collection of parameters for a DSGE model."""

    def __init__(self):
        self._params: Dict[str, Parameter] = {}

    def add(self, param: Parameter):
        """Add a parameter to the set."""
        self._params[param.name] = param

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        return self._params[name]

    def __getitem__(self, name: str) -> float:
        """Get parameter value."""
        return self._params[name].value

    def __setitem__(self, name: str, value: float):
        """Set parameter value."""
        self._params[name].value = value

    def get_values(self) -> np.ndarray:
        """Get all parameter values as array."""
        return np.array([p.value for p in self._params.values()])

    def set_values(self, values: np.ndarray):
        """Set all parameter values from array."""
        for p, v in zip(self._params.values(), values):
            if not p.fixed:
                p.value = v

    def get_free_params(self) -> Dict[str, Parameter]:
        """Get parameters that are not fixed."""
        return {name: p for name, p in self._params.items() if not p.fixed}

    def log_prior(self, values: Optional[np.ndarray] = None) -> float:
        """Evaluate log prior density."""
        if values is not None:
            self.set_values(values)

        log_prob = 0.0
        for param in self._params.values():
            if not param.fixed and param.prior is not None:
                log_prob += param.prior.logpdf(param.value)
        return log_prob

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self._params)

    def __iter__(self):
        """Iterate over parameters."""
        return iter(self._params.values())

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of name: value pairs."""
        return {name: p.value for name, p in self._params.items()}
