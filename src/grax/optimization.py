"""Optimization of parameters."""

from __future__ import annotations

import dataclasses
import enum
import warnings
from collections.abc import Callable
from typing import Generic
from typing import TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax

from grax import base


# Consecutive line search failures after which `minimize` gives up.
_MAX_LINESEARCH_FAILURES = 2

Params = TypeVar("Params")


class Status(enum.Enum):
  """The reason `minimize` stopped.

  Attributes:
    CONVERGED: the gradient norm at the returned parameters is below `tol`.
    MAX_ITER: the maximum number of iterations was reached.
    LINE_SEARCH_FAILED: the line search failed repeatedly to find a decrease.
  """

  CONVERGED = enum.auto()
  MAX_ITER = enum.auto()
  LINE_SEARCH_FAILED = enum.auto()


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class MinimizeResult(Generic[Params]):
  """The result of `minimize`.

  Attributes:
    params: the final parameters.
    n_iter: the number of iterations run, including any whose step was
      discarded because the line search failed.
    grad_norm: the norm of the objective's gradient at the final parameters.
    status: the reason optimization stopped.
  """

  params: Params
  n_iter: int
  grad_norm: float
  status: Status


@base.typed
def minimize(
  objective: Callable[[Params], jt.Float[jt.Array, ""]],
  params: Params,
  *,
  max_iter: int = 100,
  tol: float = 1e-3,
) -> MinimizeResult[Params]:
  """Minimize an objective with L-BFGS.

  Roughly follows the `run_opt` pattern from:
  https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html.

  Like L-BFGS-B, if the line search fails to find a step with sufficient
  decrease, the step is discarded and L-BFGS's memory is reset (a failure
  usually means its curvature estimates are corrupted). If the line search
  fails again straight away, this stops and warns, returning the last
  parameters that did decrease the objective.

  Args:
    objective: the function to minimize; it must be differentiable by JAX.
    params: the initial parameters, which can be any pytree.
    max_iter: the maximum number of L-BFGS iterations to run.
    tol: stop early once the gradient norm falls below this tolerance.

  Returns:
    The result of the optimization, including the reason it stopped.
  """
  # This optimizes over a flattened list of `params`'s leaves rather than
  # `params` directly. This is due to the fact that L-BFGS stacks the parameter
  # structure with an extra per-leaf "history" dimension for its internal state
  # and as a result any runtime shape checks on `params` will fail.
  flat_params, treedef = jax.tree_util.tree_flatten(params)

  def flat_objective(flat_params: list[jax.Array]) -> jt.Float[jt.Array, ""]:
    return objective(jax.tree_util.tree_unflatten(treedef, flat_params))

  optimizer = optax.lbfgs()
  value_and_grad_fun = optax.value_and_grad_from_state(flat_objective)
  init_state = optimizer.init(flat_params)

  # The carry is (params, optimizer state, iterations, consecutive failures).
  # Iterations are counted separately since a reset also resets optax's own
  # count.
  def step(
    carry: tuple[list[jax.Array], optax.OptState, jax.Array, jax.Array],
  ) -> tuple[list[jax.Array], optax.OptState, jax.Array, jax.Array]:
    flat_params, state, n_iter, n_failed = carry
    value, grad = value_and_grad_fun(flat_params, state=state)
    updates, new_state = optimizer.update(
      grad,
      state,
      flat_params,
      value=value,
      grad=grad,
      value_fn=flat_objective,
    )
    new_flat_params = optax.apply_updates(flat_params, updates)

    # When the line search fails, optax still takes an "unsafe" step that may
    # increase the objective (or none at all). Discard it and reset the
    # optimizer's memory instead. The comparison is written so that a NaN error
    # also counts as a failure.
    failed = ~(optax.tree.get(new_state, "info").decrease_error <= 0)
    flat_params = jax.tree_util.tree_map(
      lambda old, new: jnp.where(failed, old, new),
      flat_params,
      new_flat_params,
    )
    state = jax.tree_util.tree_map(
      lambda fresh, new: jnp.where(failed, fresh, new),
      init_state,
      new_state,
    )
    n_failed = jnp.where(failed, n_failed + 1, 0)
    return flat_params, state, n_iter + 1, n_failed

  def continuing_criterion(
    carry: tuple[list[jax.Array], optax.OptState, jax.Array, jax.Array],
  ) -> jt.Bool[jt.Array, ""]:
    _, state, n_iter, n_failed = carry
    # A freshly initialized (or reset) state has no gradient yet.
    is_fresh = optax.tree.get(state, "count") == 0
    grad = optax.tree.get(state, "grad")
    err = optax.tree.norm(grad)
    return (
      (n_failed < _MAX_LINESEARCH_FAILURES)
      & (n_iter < max_iter)
      & (is_fresh | (err >= tol))
    )

  init_carry = (
    flat_params,
    init_state,
    jnp.zeros((), dtype=int),
    jnp.zeros((), dtype=int),
  )
  final_flat_params, _, n_iter, n_failed = jax.lax.while_loop(
    continuing_criterion,
    step,
    init_carry,
  )

  n_iter = int(n_iter)
  grad_norm = float(
    optax.tree.norm(jax.grad(flat_objective)(final_flat_params))
  )
  converged = grad_norm < tol
  if n_failed >= _MAX_LINESEARCH_FAILURES and not converged:
    status = Status.LINE_SEARCH_FAILED
    msg = (
      f"L-BFGS stopped after {n_iter} iterations because the line search "
      f"failed {_MAX_LINESEARCH_FAILURES} times in a row (gradient norm "
      f"{grad_norm:.3g} >= tol {tol}). Returning the last parameters that "
      "decreased the objective."
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
  elif n_iter >= max_iter and not converged:
    status = Status.MAX_ITER
  else:
    status = Status.CONVERGED

  return MinimizeResult(
    params=jax.tree_util.tree_unflatten(treedef, final_flat_params),
    n_iter=n_iter,
    grad_norm=grad_norm,
    status=status,
  )
