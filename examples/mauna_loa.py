"""GP regression on the Mauna Loa atmospheric CO2 data."""

import json
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping as jt
import matplotlib.pyplot as plt
import polars as pl

from grax import gp
from grax import kernels
from grax import means


# The kernel matrix is too ill-conditioned for float32.
jax.config.update("jax_enable_x64", val=True)

# Where downloaded datasets are cached.
CACHE_DIR = Path.home() / ".cache" / "grax" / "openml"


def fetch_openml(dataset_id: int) -> pl.DataFrame:
  """Fetch a dataset from OpenML, caching it on disk.

  Datasets are cached in `CACHE_DIR`, and only downloaded if not already there.

  Args:
    dataset_id: the OpenML id of the dataset.

  Returns:
    The dataset.
  """
  path = CACHE_DIR / f"dataset_{dataset_id}.parquet"
  if path.exists():
    return pl.read_parquet(path)

  url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
  with urllib.request.urlopen(url) as response:
    info = json.load(response)["data_set_description"]
  df = pl.read_parquet(info["parquet_url"])

  # Write to a temporary file first so an interrupted write can't leave a
  # corrupt cache entry.
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_suffix(".tmp")
  df.write_parquet(temporary_path)
  temporary_path.replace(path)
  return df


def load_co2() -> pl.DataFrame:
  """Load the monthly average CO2 concentration at Mauna Loa.

  Returns:
    A frame with the time `t`, in decimal years, and the monthly average CO2
    concentration `co2`, in ppm.
  """
  weekly = fetch_openml(41187).select(
    pl.date(pl.col("year").cast(pl.Int32), "month", "day"),
    "co2",
  )
  return (
    weekly.sort("date")
    .group_by_dynamic("date", every="1mo")
    .agg(pl.col("co2").mean())
    .drop_nulls()
    .select(
      t=pl.col("date").dt.year() + pl.col("date").dt.month() / 12,
      co2="co2",
    )
  )


def plot_gp(
  tstar: jt.Float[jt.Array, " m"],
  mu: jt.Float[jt.Array, " m"],
  s2: jt.Float[jt.Array, " m"],
  *,
  data: tuple[jt.Float[jt.Array, " n"], jt.Float[jt.Array, " n"]],
) -> plt.Axes:
  """Plot a GP's posterior alongside the observed data.

  Args:
    tstar: the times the posterior was predicted at.
    mu: the posterior mean at `tstar`.
    s2: the posterior variance at `tstar`.
    data: the observed times and outputs, as `(t, y)`.

  Returns:
    The axes containing the plot.
  """
  t, y = data
  std = jnp.sqrt(s2)
  _, ax = plt.subplots()
  ax.scatter(t, y, s=6, color="black", label="Observed data")
  ax.plot(tstar, mu, label="Posterior mean")
  ax.fill_between(
    tstar,
    mu - 1.96 * std,
    mu + 1.96 * std,
    alpha=0.2,
    label="95% CI",
  )
  ax.legend()
  ax.set_title("Monthly average of air samples measurements\nfrom Mauna Loa")
  ax.set_xlabel("Year")
  ax.set_ylabel("Monthly average of CO$_2$ concentration (ppm)")
  return ax


def main() -> None:
  """Run the example."""
  co2 = load_co2()
  t = co2["t"].to_jax()
  y = co2["co2"].to_jax()

  # The following kernel follows that from scikit-learn which itself follows
  # that of Rasmussen and Williams.
  long_term_trend = 50.0**2 * kernels.SEKernel(dim=1, ell=50.0)
  seasonal = (
    2.0**2
    * kernels.SEKernel(dim=1, ell=100.0)
    * kernels.PeriodicKernel(ell=1.0, period=1.0)
  )
  irregularities = 0.5**2 * kernels.RQKernel(dim=1, ell=1.0, alpha=1.0)
  correlated_noise = 0.1**2 * kernels.SEKernel(dim=1, ell=0.1)

  # Construct the kernel, mean, and GP model.
  kernel = long_term_trend + seasonal + irregularities + correlated_noise
  mean = means.ConstantMean(input_shape=kernel.shape, offset=jnp.mean(y))
  model = gp.GP(kernel, mean, sn2=0.1**2)

  # Add data and fit the model.
  model.add_data(t[:, None], y)
  model.fit(max_iter=500)

  tstar = jnp.linspace(t.min(), 2030.0, 1000)
  mu, s2 = model.predict(tstar[:, None])

  plot_gp(tstar, mu, s2, data=(t, y))
  plt.show()


if __name__ == "__main__":
  main()
