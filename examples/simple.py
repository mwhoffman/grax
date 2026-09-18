"""A simple example of GP regression."""

import jax
import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt

from grax import gp
from grax import kernels
from grax import means


def f(x: jax.Array) -> jax.Array:
  """The latent function we're modeling."""
  return 12 + x * jnp.sin(x)


def main() -> None:
  """Run the example."""
  key = jrd.key(0)
  key, x_key, noise_key = jrd.split(key, 3)

  x = jrd.uniform(x_key, (100,), minval=0.0, maxval=10.0)
  y = f(x) + jrd.normal(noise_key, (100,))
  x = x[:, None]

  kernel = kernels.SEKernel(dim=1, rho=3.0, ell=jnp.array([0.8]))
  mean = means.ConstantMean(dim=1)
  model = gp.GP(kernel, mean, sn2=1.0)
  model.add_data(x, y)
  model.fit()

  xstar = jnp.linspace(-2.0, 12.0, 1000)
  mu, s2 = model.predict(xstar[:, None])
  std = jnp.sqrt(s2)

  plt.plot(xstar, f(xstar), linestyle="dotted", label="Latent function")
  plt.scatter(x[:, 0], y, label="Observed data")
  plt.plot(xstar, mu, label="Posterior mean")
  plt.fill_between(
    xstar,
    mu - 1.96 * std,
    mu + 1.96 * std,
    alpha=0.2,
    label="95% CI",
  )
  plt.legend()
  plt.xlim(-2, 12)
  plt.title("Gaussian process regression")
  plt.xlabel("Inputs, x")
  plt.ylabel("Outputs, y")
  plt.show()


if __name__ == "__main__":
  main()
