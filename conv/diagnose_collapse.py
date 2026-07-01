"""Architecture scan: instrumented Conv-HGF training to localise misalignment with PCX.

Runs the same VGG5 setup as conv_hgf_cifar.py but prints a compact per-layer
stat table after every epoch.  Stats are read directly from the carry state
(net.state.layers) left by the last training sample — zero overhead, no extra
forward pass, no JIT retrace.

Per-layer columns:
  |μ̂|     — mean |expected_mean| (activation magnitude)
  std(μ̂)  — std of expected_mean (activation diversity; ~0 = collapsed)
  |ΔM|    — mean |mean - expected_mean|
             conv/FC layers: the posterior step size (how much the single
               gradient step moved the hidden state).  Near-zero = step is
               doing nothing.  Very large = likely overshooting.
             output layer: mean prediction error.
  prec    — mean posterior precision
  prec_max— max posterior precision (large → step size → 0)
  dead    — conv channels with spatial std < DEAD_THRESHOLD (gradient killers)

Weight columns per weight tensor:
  kern_norm — Frobenius norm of the conv kernel (or FC weight)
  kern_max  — abs-max element
  |bias|    — mean absolute bias (conv only)

NaN / Inf / large-value (>LARGE_THRESHOLD) scan runs every epoch and
switches to fine-grained chunk scanning near the collapse.

NOTE: this script runs without mini-batch gradient accumulation (batch_size=1,
per-sample updates).  Prior runs showed similar accuracy to batch_size=32; it
removes one variable and avoids a second JIT trace.  To test the batched path
set BATCH_SIZE env var to 32 and change the fit() calls accordingly.

Env vars:
  EPOCHS      total epochs (default 30)
  FINE_FROM   epoch to start fine-grained scanning (default 23)
  FINE_CHUNK  samples per fine-grained chunk (default 1600)
  USE_FC      1 (default) or 0
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import optax

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils_cifar import load_cifar10, augment
from pyhgf.model import DeepNetwork

TONIC_VOL     = -10.0
TONIC_VOL_VOL = -10.0
BASE_LR       = 0.0001
LEARNING_KIND = "standard"
SEED          = 0
USE_FC        = os.environ.get("USE_FC", "1") != "0"

EPOCHS     = int(os.environ.get("EPOCHS", "30"))
FINE_FROM  = int(os.environ.get("FINE_FROM", "23"))
FINE_CHUNK = int(os.environ.get("FINE_CHUNK", "1600"))

DEAD_THRESHOLD  = 1e-3   # spatial std below which a conv channel is "dead"
LARGE_THRESHOLD = 1e4    # early-warning: flag before values hit NaN

leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)

X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)
Y_tr = np.eye(10, dtype=np.float32)[y_tr]


def build_network(seed):
    net = DeepNetwork(coupling_fn=leaky_relu)
    net.add_layer(size=10, kind="binary",
                  tonic_volatility=TONIC_VOL,
                  tonic_volatility_vol=TONIC_VOL_VOL,
                  add_constant_input=False,
                  volatility_parent=False)
    if USE_FC:
        net.add_layer(size=512,
                      tonic_volatility=TONIC_VOL,
                      tonic_volatility_vol=TONIC_VOL_VOL,
                      add_constant_input=True,
                      volatility_parent=False)
    net.add_conv_layer(out_channels=512,
                       kernel_size=3,
                       padding="VALID",
                       pool=True,
                       tonic_volatility=TONIC_VOL,
                       tonic_volatility_vol=TONIC_VOL_VOL)
    for out_ch in [512, 256, 128]:
        net.add_conv_layer(out_channels=out_ch,
                           kernel_size=3,
                           pool=True,
                           tonic_volatility=TONIC_VOL,
                           tonic_volatility_vol=TONIC_VOL_VOL)
    net.add_spatial_input(C=3, H=32, W=32)
    net.weight_initialisation("he", key=jax.random.key(seed))
    return net


def evaluate(net):
    preds = np.array(net.predict(jnp.array(X_te)))
    return 100.0 * (np.argmax(preds, axis=1) == y_te).mean()


# ── Per-layer stat table ───────────────────────────────────────────────────────

def _arr_stats(arr):
    """(mean_abs, std, mean_prec, max_prec) from a JAX/np array."""
    a = np.asarray(arr)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(finite))), float(np.std(finite))


def print_epoch_stats(net, epoch, acc, elapsed):
    """Read carry state and print compact per-layer + per-weight table."""
    state = net.state
    kinds = net.layer_kinds
    n     = len(state.layers)

    print(f"\n=== Epoch {epoch:>3} | acc={acc:.2f}% | {elapsed:.1f}s ===")
    print(f"  {'L':>2}  {'kind':>8}  {'|μ̂|':>7}  {'std(μ̂)':>7}  "
          f"{'|ΔM|':>9}  {'prec':>7}  {'prec_max':>9}  dead")

    for li in range(n - 1, -1, -1):   # input → output
        layer = state.layers[li]
        kind  = kinds[li]

        em  = np.asarray(layer.expected_mean)
        m   = np.asarray(layer.mean)
        pr  = np.asarray(layer.precision)

        em_abs, em_std = _arr_stats(em)
        delta_m        = float(np.mean(np.abs(m - em)))

        pr_fin   = pr[np.isfinite(pr)]
        prec_m   = float(np.mean(pr_fin))  if pr_fin.size else float("nan")
        prec_max = float(np.max(pr_fin))   if pr_fin.size else float("nan")

        # Dead-channel count for conv layers with spatial extent > 1×1.
        # For 1×1 spatial (e.g. the bottleneck VALID layer) std is trivially 0
        # so the metric is meaningless — show "-" instead.
        dead_str = "   -"
        if em.ndim == 3 and em.shape[1] > 1 and em.shape[2] > 1:
            ch_stds  = np.std(em, axis=(1, 2))
            n_dead   = int((ch_stds < DEAD_THRESHOLD).sum())
            dead_str = f"{n_dead:>3}/{em.shape[0]}"

        # Flag anomalies inline (skip epoch 0: carry state is all-zeros before training)
        flag = ""
        if epoch > 0:
            if kind not in ("binary",) and delta_m < 1e-6:
                flag = " [~0 step]"
            elif delta_m > 50.0:
                flag = " [LARGE]"
            if prec_max > 1e6:
                flag += " [prec!]"

        print(f"  {li:>2}  {kind:>8}  {em_abs:>7.4f}  {em_std:>7.4f}  "
              f"{delta_m:>9.5f}  {prec_m:>7.3f}  {prec_max:>9.3f}  "
              f"{dead_str}{flag}")

    # Weight table
    print("  Weights:")
    for wi, w in enumerate(state.weights):
        if isinstance(w, tuple):
            k = np.asarray(w[0])
            b = np.asarray(w[1])
            print(f"    W{wi} (conv):  kern_norm={np.linalg.norm(k):.4f}"
                  f"  kern_max={np.abs(k).max():.5f}"
                  f"  |bias|={np.abs(b).mean():.5f}")
        else:
            wa = np.asarray(w)
            print(f"    W{wi}   (FC):  norm={np.linalg.norm(wa):.4f}"
                  f"  absmax={np.abs(wa).max():.5f}")
    print()


# ── NaN / Inf / large-value scan ──────────────────────────────────────────────

LAYER_FIELDS = (
    "mean", "precision", "expected_mean", "expected_precision",
    "mean_vol", "precision_vol",
)


def _stat(arr):
    a = np.asarray(arr)
    finite = np.isfinite(a)
    has_nan = bool(np.isnan(a).any())
    has_inf = bool(np.isinf(a).any())
    if finite.any():
        f = a[finite]
        return has_nan, has_inf, float(f.min()), float(f.max()), float(np.abs(f).max())
    return has_nan, has_inf, float("nan"), float("nan"), float("nan")


def scan_bad(net, tag):
    """Flag NaN / Inf / > LARGE_THRESHOLD tensors.  Returns True if any found."""
    state = net.state
    kinds = net.layer_kinds
    found = False

    for li, layer in enumerate(state.layers):
        for field in LAYER_FIELDS:
            arr = getattr(layer, field)
            has_nan, has_inf, lo, hi, amax = _stat(arr)
            if has_nan or has_inf or amax > LARGE_THRESHOLD:
                found = True
                print(f"  [{tag}] L{li} ({kinds[li]}).{field}: "
                      f"nan={has_nan} inf={has_inf} "
                      f"min={lo:.4g} max={hi:.4g} absmax={amax:.4g}", flush=True)

    for wi, w in enumerate(state.weights):
        tensors = w if isinstance(w, tuple) else (w,)
        names   = ("kernel", "bias") if isinstance(w, tuple) else ("weight",)
        for name, arr in zip(names, tensors):
            has_nan, has_inf, lo, hi, amax = _stat(arr)
            if has_nan or has_inf or amax > LARGE_THRESHOLD:
                found = True
                print(f"  [{tag}] W{wi}.{name}: "
                      f"nan={has_nan} inf={has_inf} "
                      f"min={lo:.4g} max={hi:.4g} absmax={amax:.4g}", flush=True)

    return found


# ── Main ──────────────────────────────────────────────────────────────────────

# fit() compares the optimizer by identity to decide whether to reinit
# opt_state, so the same instance must be reused across every fit() call.
OPTIMIZER = optax.sgd(BASE_LR)

print(f"LR={BASE_LR} (no batching, per-sample updates)  use_fc={USE_FC}  "
      f"EPOCHS={EPOCHS}  FINE_FROM={FINE_FROM}  FINE_CHUNK={FINE_CHUNK}", flush=True)
print("Devices:", jax.devices(), flush=True)
print("JIT warm-up...", flush=True)
_net = build_network(seed=0)
_net.fit(X_tr[:4], Y_tr[:4], optimizer=OPTIMIZER, learning_kind=LEARNING_KIND)
print("Done.\n", flush=True)

rng = np.random.default_rng(SEED)
net = build_network(SEED)

acc0 = evaluate(net)
print(f"Epoch  0/{EPOCHS}  acc={acc0:.2f}%  (init, should be ~10%)", flush=True)
print_epoch_stats(net, 0, acc0, 0.0)

stop = False
for epoch in range(1, EPOCHS + 1):
    t0    = time.time()
    idx   = rng.permutation(len(X_tr))
    X_aug = augment(X_tr[idx], rng)
    Y_aug = Y_tr[idx]

    if epoch >= FINE_FROM:
        n = len(X_aug)
        for start in range(0, n, FINE_CHUNK):
            end = min(start + FINE_CHUNK, n)
            net.fit(X_aug[start:end], Y_aug[start:end],
                    optimizer=OPTIMIZER, learning_kind=LEARNING_KIND)
            bad = scan_bad(net, f"E{epoch}:S{end}")
            if bad:
                acc = evaluate(net)
                print(f"  --> first bad tensor at epoch {epoch}, sample {end} "
                      f"(acc={acc:.2f}%) <--", flush=True)
                stop = True
                break
    else:
        net.fit(X_aug, Y_aug, optimizer=OPTIMIZER, learning_kind=LEARNING_KIND)
        scan_bad(net, f"E{epoch}")

    acc = evaluate(net)
    dt  = time.time() - t0
    print(f"Epoch {epoch:>2}/{EPOCHS}  acc={acc:.2f}%  ({dt:.1f}s)", flush=True)
    print_epoch_stats(net, epoch, acc, dt)

    if stop:
        print("\nStopping early: instability located above.", flush=True)
        break

print("\nDone.", flush=True)
