"""Collection of likelihood functions."""

__all__ = ["Gaussian", "Likelihood"]

from grax.likelihoods.base import Likelihood
from grax.likelihoods.gaussian import Gaussian
