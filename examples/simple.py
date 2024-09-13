"""Simple example."""

import matplotlib.pyplot as plt
import numpy as np

from grax import gp
from grax import kernels
from grax import likelihoods


def main():
  """Run the example."""

  # Define the function we want to model.
  def f(x: np.ndarray) -> np.ndarray:
    return x * np.sin(x)

  # Get an rng.
  rng = np.random.default_rng()

  # Draw samples from the function.
  X = rng.uniform(0, 10, size=100)
  Y = f(X) + rng.normal(size=len(X))
  X = X[:, None]

  # Create our model.
  model = gp.GP(
    kernels.SquaredExponential(3.0, 0.8, dim=1),
    likelihoods.Gaussian(1.0),
    data=(X, Y)
  )

  # Get posterior predictions.
  x = np.linspace(-2, 12, 1000)
  mu, s2 = model.predict(x[:, None])

  # Plot the samples and predictions.
  plt.gcf()
  plt.cla()
  plt.plot(x, f(x), linestyle="dotted", label="Latent function")
  plt.scatter(X.flatten(), Y, label="Observed data")
  plt.plot(x, mu, linestyle="-", label="Posterior mean")
  plt.fill_between(x, mu - 1.96 * s2, mu + 1.96 * s2, alpha=0.2, label="95% CI")
  plt.legend()
  plt.xlim(-2, 12)
  plt.title("Gaussian process regression")
  plt.xlabel("Inputs, X")
  plt.ylabel("Outputs, Y")

  # Restyle the plot.
  ax = plt.gca()
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  ax.yaxis.label.set_fontsize(16)
  [t.set_fontsize(12) for t in ax.get_xticklabels() + ax.get_yticklabels()]
  [t.set_fontsize(14) for t in ax.get_legend().get_texts()]

  # Display the plot.
  plt.show()


if __name__ == "__main__":
  main()
