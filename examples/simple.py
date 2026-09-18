"""A simple example of GP regression."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jrd
import jaxtyping as jt
import matplotlib.pyplot as plt

from grax import gp
from grax import kernels
from grax import means


def f(x: jax.Array) -> jax.Array:
  """The latent function we're modeling."""
  return 12 + x * jnp.sin(x)


def plot_posterior(
  xstar: jt.Float[jt.Array, " m"],
  mu: jt.Float[jt.Array, " m"],
  s2: jt.Float[jt.Array, " m"],
  *,
  data: tuple[jt.Float[jt.Array, " n"], jt.Float[jt.Array, " n"]],
  f: Callable[[jax.Array], jax.Array],
) -> plt.Axes:
  """Plot the posterior alongside the observed data and latent function.

  Args:
    xstar: the inputs the posterior was predicted at.
    mu: the posterior mean at `xstar`.
    s2: the posterior variance at `xstar`.
    data: the observed inputs and outputs, as `(x, y)`.
    f: the latent function, plotted for reference.

  Returns:
    The axes containing the plot.
  """
  x, y = data
  std = jnp.sqrt(s2)
  _, ax = plt.subplots()
  ax.plot(xstar, f(xstar), linestyle="dotted", label="Latent function")
  ax.scatter(x, y, label="Observed data")
  ax.plot(xstar, mu, label="Posterior mean")
  ax.fill_between(
    xstar,
    mu - 1.96 * std,
    mu + 1.96 * std,
    alpha=0.2,
    label="95% CI",
  )
  return ax


def main() -> None:
  """Run the example."""
  key = jrd.key(0)
  _, x_key, noise_key = jrd.split(key, 3)

  # Generate random data.
  x = jrd.uniform(x_key, (100,), minval=0.0, maxval=10.0)
  y = f(x) + jrd.normal(noise_key, (100,))

  # Define the model.
  kernel = 3.0 * kernels.SEKernel(dim=1, ell=0.8)
  mean = means.ConstantMean(input_shape=kernel.shape)
  model = gp.GP(kernel, mean, sn2=1.0)

  # Fit the model.
  model.add_data(x[:, None], y)
  model.fit()

  # Get predictions.
  xstar = jnp.linspace(-2.0, 12.0, 1000)
  mu, s2 = model.predict(xstar[:, None])

  # Plot everything.
  ax = plot_posterior(xstar, mu, s2, data=(x, y), f=f)
  ax.legend()
  ax.set_xlim(-2, 12)
  ax.set_title("Gaussian process regression")
  ax.set_xlabel("Inputs, x")
  ax.set_ylabel("Outputs, y")
  plt.show()


if __name__ == "__main__":
  main()
