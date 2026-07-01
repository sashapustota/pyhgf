# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Convolutional prediction and posterior update for deep predictive coding networks."""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerParams, LayerState


def vectorized_conv_prediction(
    child_state: LayerState,
    parent_state: LayerState,
    kernel: jnp.ndarray,
    bias: jnp.ndarray,
    params: LayerParams,
    time_step: float,
    coupling_fn: Callable = lambda x: x,
    stride: int = 1,
    padding: str = "SAME",
    pool: bool = False,
    pool_size: int = 2,
    pool_stride: int = 2,
) -> LayerState:
    """Predict expected mean/precision for a convolutional layer.

    Always runs in ``volatility_parent=False`` mode: only ``tonic_volatility``
    drives the expected precision.

    Parameters
    ----------
    child_state :
        Current state of the child (output) conv layer.
    parent_state :
        Current state of the parent (input) layer.
    kernel :
        Conv kernel, shape ``(out_ch, in_ch, kH, kW)``.
    bias :
        Per-channel bias, shape ``(out_ch,)``.
    params :
        Child layer parameters.
    time_step :
        Current time step.
    coupling_fn :
        Activation function applied to parent activations.
    stride :
        Conv stride.
    padding :
        ``"SAME"`` or ``"VALID"``.
    pool :
        Whether to apply max pooling after conv.
    pool_size :
        Max-pool window size.
    pool_stride :
        Max-pool stride.

    Returns
    -------
    LayerState
        Updated child state with expected mean and precision filled in.
    """
    parent_mean = coupling_fn(parent_state.expected_mean)  # (in_ch, H, W)

    # Forward conv: (1, in_ch, H, W) → (1, out_ch, H_out, W_out) → (out_ch, H_out, W_out)
    out = jax.lax.conv_general_dilated(
        parent_mean[None],
        kernel,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0]

    out = out + bias[:, None, None]

    if pool:
        out = jax.lax.reduce_window(
            out,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, pool_size, pool_size),
            window_strides=(1, pool_stride, pool_stride),
            padding="VALID",
        )

    expected_mean = out

    # Precision prediction (volatility_parent=False: only tonic_volatility)
    predicted_volatility = time_step * jnp.exp(params.tonic_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )
    expected_precision = 1.0 / (1.0 / child_state.precision + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision

    return dataclasses.replace(
        child_state,
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        # No AR-volatility chain and no value-coupling variance for conv layers
        # (volatility_parent=False, no jax.grad-through-conv Laplace term) — the
        # conditional and marginal predicted precisions coincide, matching the
        # binary-leaf / is_input_layer convention in the FC prediction path.
        conditional_expected_precision=expected_precision,
        effective_precision=effective_precision,
        expected_mean_vol=child_state.mean_vol,
        expected_precision_vol=child_state.precision_vol,
        effective_precision_vol=child_state.effective_precision_vol,
    )


def vectorized_conv_parent_posterior_from_conv(
    parent_state: LayerState,
    child_state: LayerState,
    kernel: jnp.ndarray,
    bias: jnp.ndarray,
    coupling_fn: Callable,
    stride: int,
    padding: str,
    pool: bool,
    pool_size: int,
    pool_stride: int,
) -> LayerState:
    """Gradient-based posterior update for a conv parent from a conv child.

    One step of gradient ascent on the free energy w.r.t. the parent's
    activations. The gradient is computed via :func:`jax.grad` on the proxy
    loss, which is equivalent to a single ``conv_transpose`` backprop step.

    Parameters
    ----------
    parent_state :
        Current state of the conv parent layer.
    child_state :
        Current state of the conv child layer (must have ``mean`` set).
    kernel :
        Conv kernel, shape ``(out_ch, in_ch, kH, kW)``.
    bias :
        Per-channel bias, shape ``(out_ch,)``.
    coupling_fn :
        Activation function applied to parent activations.
    stride, padding, pool, pool_size, pool_stride :
        Conv and pooling hyperparameters (same as :func:`vectorized_conv_prediction`).

    Returns
    -------
    LayerState
        Parent state with updated ``mean`` (posterior activation).
    """
    pi_child = child_state.expected_precision

    def _free_energy(parent_mean):
        out = jax.lax.conv_general_dilated(
            coupling_fn(parent_mean)[None],
            kernel,
            window_strides=(stride, stride),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0]
        out = out + bias[:, None, None]
        if pool:
            out = jax.lax.reduce_window(
                out,
                init_value=-jnp.inf,
                computation=jax.lax.max,
                window_dimensions=(1, pool_size, pool_size),
                window_strides=(1, pool_stride, pool_stride),
                padding="VALID",
            )
        pe = child_state.mean - out
        return -0.5 * jnp.sum(pe**2 * pi_child)

    grad = jax.grad(_free_energy)(parent_state.expected_mean)

    # One gradient step.  Step size = 1/expected_precision, clamped to ≤1.0.
    # Without the clamp, precision decay (driven by large PEs early in training)
    # causes the step size to grow to ~50×, amplifying gradients and driving
    # runaway kernel growth.  The clamp keeps the step at its initial value (1.0
    # when precision=1.0) and only reduces it if precision grows above 1.
    step_size = jnp.clip(
        1.0 / (parent_state.expected_precision + 1e-8), 0.0, 1.0
    )
    posterior_mean = parent_state.expected_mean + step_size * grad

    return dataclasses.replace(
        parent_state,
        mean=posterior_mean,
        precision=parent_state.expected_precision,
    )


def vectorized_conv_parent_posterior_from_fc(
    parent_state: LayerState,
    child_state: LayerState,
    fc_weights: jnp.ndarray,
    coupling_fn: Callable,
    add_constant_input: bool,
) -> LayerState:
    """Gradient-based posterior update for a conv parent from an FC child.

    Flattens the conv parent activations and uses :func:`jax.grad` on the
    free energy proxy, which reduces to a matrix-transpose backprop step.

    Parameters
    ----------
    parent_state :
        Current state of the conv parent layer.
    child_state :
        Current state of the FC child layer.
    fc_weights :
        FC weight matrix, shape ``(fc_size, flat_conv)`` or
        ``(fc_size, flat_conv + 1)`` with bias column.
    coupling_fn :
        Activation function applied to parent activations.
    add_constant_input :
        Whether *fc_weights* has a bias column (constant input).

    Returns
    -------
    LayerState
        Parent state with updated ``mean``.
    """
    pi_child = child_state.expected_precision

    def _free_energy(parent_mean):
        flat = parent_mean.ravel()
        if add_constant_input:
            flat = jnp.concatenate([flat, jnp.ones(1)])
        expected_child = jnp.matmul(fc_weights, coupling_fn(flat))
        pe = child_state.mean - expected_child
        return -0.5 * jnp.sum(pe**2 * pi_child)

    grad = jax.grad(_free_energy)(parent_state.expected_mean)

    step_size = jnp.clip(
        1.0 / (parent_state.expected_precision + 1e-8), 0.0, 1.0
    )
    posterior_mean = parent_state.expected_mean + step_size * grad

    return dataclasses.replace(
        parent_state,
        mean=posterior_mean,
        precision=parent_state.expected_precision,
    )
