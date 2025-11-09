"""Tests for model specification."""

import numpy as np
import pytest
from dsge import Parameter, ParameterSet, Prior, ModelSpecification


def test_prior_normal():
    """Test normal prior."""
    prior = Prior('normal', {'mean': 0.0, 'std': 1.0})
    samples = prior.rvs(1000)
    assert np.abs(samples.mean()) < 0.1
    assert np.abs(samples.std() - 1.0) < 0.1


def test_prior_beta():
    """Test beta prior."""
    prior = Prior('beta', {'alpha': 2.0, 'beta': 2.0})
    samples = prior.rvs(1000)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)
    assert np.abs(samples.mean() - 0.5) < 0.1


def test_parameter():
    """Test parameter creation."""
    param = Parameter(
        name='alpha',
        value=0.33,
        bounds=(0, 1),
        prior=Prior('beta', {'alpha': 3.3, 'beta': 6.7})
    )
    assert param.name == 'alpha'
    assert param.value == 0.33


def test_parameter_bounds():
    """Test parameter bounds validation."""
    with pytest.raises(ValueError):
        Parameter(name='alpha', value=1.5, bounds=(0, 1))


def test_parameter_set():
    """Test parameter set."""
    params = ParameterSet()
    params.add(Parameter(name='alpha', value=0.33))
    params.add(Parameter(name='beta', value=0.99))

    assert len(params) == 2
    assert params['alpha'] == 0.33
    assert params['beta'] == 0.99

    # Test value setting
    params['alpha'] = 0.35
    assert params['alpha'] == 0.35


def test_parameter_set_log_prior():
    """Test log prior evaluation."""
    params = ParameterSet()
    params.add(Parameter(
        name='alpha',
        value=0.5,
        prior=Prior('beta', {'alpha': 2.0, 'beta': 2.0})
    ))

    log_prior = params.log_prior()
    assert np.isfinite(log_prior)


def test_model_specification():
    """Test model specification."""
    spec = ModelSpecification(
        n_states=2,
        n_controls=3,
        n_shocks=1,
        n_observables=2,
        state_names=['k', 'z'],
        control_names=['c', 'n', 'y'],
        shock_names=['eps'],
        observable_names=['y_obs', 'n_obs']
    )

    assert spec.n_states == 2
    assert spec.n_controls == 3
    assert len(spec.state_names) == 2
    assert len(spec.control_names) == 3
