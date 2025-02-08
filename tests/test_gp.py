"""Tests for the GP."""

import numpy as np
import pytest
import testing

from grax import gp
from grax import kernels
from grax import likelihoods
from grax import means
from grax import types
from grax.utils import checks


def test_init():
  kernel = kernels.SquaredExponential(1.0, 1.0, dim=1)
  likelihood = likelihoods.Gaussian(1.0)
  mean = means.Zero(dim=1)
  data = (np.zeros((5, 1)), np.zeros(5))

  gp.GP(kernel, likelihood, mean)
  gp.GP(kernel, likelihood, mean, data)

  with pytest.raises(checks.CheckError):
    gp.GP(kernel, likelihood, means.Zero(dim=2))


def test_add_data():
  rng = np.random.default_rng()

  for dim in range(1, 3+1):
    # Initialize a GP model.
    kernel = kernels.SquaredExponential(1.0, 1.0, dim=dim)
    likelihood = likelihoods.Gaussian(1.0)
    mean = means.Zero(dim=dim)
    model = gp.GP(kernel, likelihood, mean)

    # Make sure we don't compute stats with no data.
    assert model._get_stats() is None

    # Ensure we can add initial data.
    X = rng.uniform(size=(5, dim))
    Y = rng.normal(size=len(X))
    model.add_data(X, Y)

    # Ensure we can add additional data.
    X = rng.uniform(size=(5, dim))
    Y = rng.normal(size=len(X))
    model.add_data(X, Y)

    # Adding data should fail if the input dimensions are wrong.
    X = rng.uniform(size=(5, dim+1))
    Y = rng.normal(size=len(X))
    with pytest.raises(checks.CheckError):
      model.add_data(X, Y)


def test_predict():
  rng = np.random.default_rng()

  for dim in range(1, 3+1):
    # Initialize a GP model.
    kernel = kernels.SquaredExponential(1.0, 1.0, dim=dim)
    likelihood = likelihoods.Gaussian(1.0)
    mean = means.Zero(dim=dim)
    model = gp.GP(kernel, likelihood, mean)

    # Ensure we can make prior predictions.
    X = rng.uniform(size=(10, dim))
    mu, s2 = model.predict(X)
    assert mu.shape == (len(X),)
    assert s2.shape == (len(X),)

    # Add data.
    X = rng.uniform(size=(5, dim))
    Y = rng.normal(size=len(X))
    model.add_data(X, Y)

    # Ensure we can make posterior predictions.
    X = rng.uniform(size=(10, dim))
    mu, s2 = model.predict(X)
    assert mu.shape == (len(X),)
    assert s2.shape == (len(X),)

    # Make predictions again which should re-use the cache.
    X = rng.uniform(size=(10, dim))
    mu, s2 = model.predict(X)
    assert mu.shape == (len(X),)
    assert s2.shape == (len(X),)

    # Predictions should fail if the input dimensions are wrong.
    X = rng.uniform(size=(10, dim+1))
    with pytest.raises(checks.CheckError):
      model.predict(X)


def test_log_likelihood():
  rng = np.random.default_rng()

  for dim in range(1, 3+1):
    # Initialize a GP model.
    kernel = kernels.SquaredExponential(1.0, 1.0, dim=dim)
    likelihood = likelihoods.Gaussian(1.0)
    mean = means.Zero(dim=dim)
    model = gp.GP(kernel, likelihood, mean)

    assert model.log_likelihood() == 0.0

    # Add data.
    X = rng.uniform(size=(5, dim))
    Y = rng.normal(size=len(X))
    model.add_data(X, Y)

    # Evaluate the log-likelihood.
    model.log_likelihood()


@testing.parameterize_goldens(
  dict(
    kernel=kernels.SquaredExponential(1.0, 1.0, dim=1),
    likelihood=likelihoods.Gaussian(1.0),
    X=np.linspace(0, 1, 10)[:, None],
    Y=np.zeros(10),
    Z=np.linspace(0, 1, 100)[:, None],
  ),
)
def test_golden_predict(
  kernel, likelihood, X, Y, Z
) -> tuple[types.Array, types.Array]:
  return gp.GP(kernel, likelihood, data=(X, Y)).predict(Z)


@testing.parameterize_goldens(
  dict(
    kernel=kernels.SquaredExponential(1.0, 1.0, dim=1),
    likelihood=likelihoods.Gaussian(1.0),
    X=np.linspace(0, 1, 10)[:, None],
    Y=np.zeros(10),
  ),
)
def test_golden_log_likelihood(
  kernel, likelihood, X, Y
) -> types.Array:
  return gp.GP(kernel, likelihood, data=(X, Y)).log_likelihood()
