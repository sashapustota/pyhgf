# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Convolutional weight gradient for deep predictive coding networks.

Only computes the raw descent gradient — matching
:func:`pyhgf.updates.vectorized.learning.vectorized_weight_gradient`'s calling
convention and sign. Applying the gradient (Adam/SGD/etc.) is handled generically
by the ``optax`` optimizer in :func:`pyhgf.utils.vectorized_belief_propagation.
propagation_step`, the same as every other layer kind.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerState


def vectorized_conv_weight_gradient(
    parent_state: LayerState,
    child_state: LayerState,
    kernel: jnp.ndarray,
    bias: jnp.ndarray,
    coupling_fn: Callable,
    stride: int = 1,
    padding: str = "SAME",
    pool: bool = False,
    pool_size: int = 2,
    pool_stride: int = 2,
    kind: str = "precision_weighted",
) -> tuple:
    """Descent gradient for a conv kernel/bias connecting a conv child to a conv parent.

    The gradient w.r.t. the kernel/bias is computed using :func:`jax.grad` on a
    proxy free-energy term, which is equivalent to the standard cross-correlation
    backward pass and correctly handles pooling. Sign-flipped to a descent
    gradient, mirroring :func:`~pyhgf.updates.vectorized.learning.
    vectorized_weight_gradient` so it composes with ``optax``
    (``apply_updates(weights, updates)`` performs ``weights + updates``).

    Parameters
    ----------
    parent_state :
        State of the conv/spatial parent (input feature maps).
    child_state :
        State of the conv child (output feature maps).
    kernel :
        Current kernel, shape ``(out_ch, in_ch, kH, kW)``.
    bias :
        Current bias, shape ``(out_ch,)``.
    coupling_fn :
        Activation applied to parent activations before conv.
    stride, padding, pool, pool_size, pool_stride :
        Conv and pooling hyperparameters.
    kind :
        Gradient mode: ``"precision_weighted"`` (default) weights the
        prediction error by the child's posterior precision; anything else
        falls back to the unweighted ``"standard"`` gradient. (Conv doesn't yet
        support the ``precision_ratio``/``map_natural``/``pure_natural`` modes
        that the generic FC gradient does.)

    Returns
    -------
    grad :
        Descent gradient tuple ``(grad_kernel, grad_bias)``, same shapes as
        ``(kernel, bias)``. NaN/inf entries are zeroed so optax's moment
        accumulators stay finite.
    """
    pe = child_state.mean - child_state.expected_mean
    pe_weighted = pe * child_state.precision if kind == "precision_weighted" else pe
    parent_acts = coupling_fn(parent_state.mean)

    def _proxy(k, b):
        out = jax.lax.conv_general_dilated(
            parent_acts[None],
            k,
            window_strides=(stride, stride),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0]
        out = out + b[:, None, None]
        if pool:
            out = jax.lax.reduce_window(
                out,
                init_value=-jnp.inf,
                computation=jax.lax.max,
                window_dimensions=(1, pool_size, pool_size),
                window_strides=(1, pool_stride, pool_stride),
                padding="VALID",
            )
        return jnp.sum(out * pe_weighted)

    grad_k, grad_b = jax.grad(_proxy, argnums=(0, 1))(kernel, bias)

    grad_k = jnp.where(jnp.isnan(grad_k) | jnp.isinf(grad_k), 0.0, grad_k)
    grad_b = jnp.where(jnp.isnan(grad_b) | jnp.isinf(grad_b), 0.0, grad_b)

    return (-grad_k, -grad_b)
