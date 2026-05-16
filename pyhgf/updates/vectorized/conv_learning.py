# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Convolutional weight update for deep predictive coding networks."""

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

from pyhgf.typing import LayerState


def vectorized_conv_weight_update(
    child_state: LayerState,
    parent_state: LayerState,
    kernel: jnp.ndarray,
    bias: jnp.ndarray,
    coupling_fn: Callable,
    lr: Union[float, str],
    stride: int = 1,
    padding: str = "SAME",
    pool: bool = False,
    pool_size: int = 2,
    pool_stride: int = 2,
    kind: str = "precision_weighted",
    adam_m: Optional[tuple] = None,
    adam_v: Optional[tuple] = None,
    adam_t: int = 0,
    adam_lr: float = 1e-3,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> tuple:
    """Update conv kernel and bias via a proxy-loss gradient.

    The gradient w.r.t. the kernel is computed using :func:`jax.grad` on a
    proxy free-energy term, which is equivalent to the standard cross-correlation
    backward pass and correctly handles pooling.

    Parameters
    ----------
    child_state :
        State of the conv child (output feature maps).
    parent_state :
        State of the conv/spatial parent (input feature maps).
    kernel :
        Current kernel, shape ``(out_ch, in_ch, kH, kW)``.
    bias :
        Current bias, shape ``(out_ch,)``.
    coupling_fn :
        Activation applied to parent activations before conv.
    lr :
        ``float`` for direct gradient scaling, or ``"adam"`` for the Adam
        optimiser.
    stride, padding, pool, pool_size, pool_stride :
        Conv and pooling hyperparameters.
    kind :
        Gradient mode: ``"standard"``, ``"precision_weighted"`` (default),
        or ``"precision_ratio"``.
    adam_m :
        Adam first-moment tuple ``(kernel_m, bias_m)``. Required when
        ``lr="adam"``.
    adam_v :
        Adam second-moment tuple ``(kernel_v, bias_v)``.
    adam_t :
        Global Adam timestep (pre-incremented).
    adam_lr, adam_beta1, adam_beta2, adam_epsilon :
        Adam hyper-parameters.

    Returns
    -------
    new_weights :
        Tuple ``(new_kernel, new_bias)``.
    new_adam_m :
        Updated Adam first moments (or ``None``).
    new_adam_v :
        Updated Adam second moments (or ``None``).
    """
    pe = child_state.mean - child_state.expected_mean  # (out_ch, H_out, W_out)

    if kind == "precision_weighted":
        pe_weighted = pe * child_state.precision
    else:
        pe_weighted = pe  # standard

    parent_acts = coupling_fn(parent_state.mean)  # (in_ch, H_in, W_in)

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

    if lr == "adam":
        assert adam_m is not None and adam_v is not None
        km, bm = adam_m
        kv, bv = adam_v

        new_km = adam_beta1 * km + (1.0 - adam_beta1) * grad_k
        new_kv = adam_beta2 * kv + (1.0 - adam_beta2) * grad_k**2
        km_hat = new_km / (1.0 - adam_beta1**adam_t)
        kv_hat = new_kv / (1.0 - adam_beta2**adam_t)
        dk = adam_lr * km_hat / (jnp.sqrt(kv_hat) + adam_epsilon)

        new_bm = adam_beta1 * bm + (1.0 - adam_beta1) * grad_b
        new_bv = adam_beta2 * bv + (1.0 - adam_beta2) * grad_b**2
        bm_hat = new_bm / (1.0 - adam_beta1**adam_t)
        bv_hat = new_bv / (1.0 - adam_beta2**adam_t)
        db = adam_lr * bm_hat / (jnp.sqrt(bv_hat) + adam_epsilon)

        new_m = (new_km, new_bm)
        new_v = (new_kv, new_bv)
    else:
        dk = float(lr) * grad_k
        db = float(lr) * grad_b
        new_m = None
        new_v = None

    dk = jnp.where(jnp.isnan(dk) | jnp.isinf(dk), 0.0, dk)
    db = jnp.where(jnp.isnan(db) | jnp.isinf(db), 0.0, db)

    new_kernel = kernel + dk
    new_bias = bias + db

    return (new_kernel, new_bias), new_m, new_v
