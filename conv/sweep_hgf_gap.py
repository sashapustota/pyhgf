"""Sweep: isolate each gap between Conv-HGF (80.8%) and PCX (86-89%).

Changes tested individually and all-together (30 epochs each):
  arch   – SAME padding on last conv + no FC layer (matches PCX 2×2 spatial head)
  bs128  – batch_size 32 → 128
  adamw  – post-epoch weight decay 1.2e-5 (approximates AdamW)
  cosine – cosine LR warmup/decay schedule (epoch-level)
  gelu   – GELU activation instead of leaky_relu
  all    – all five changes combined

Results → conv/results/sweep_hgf_gap_<name>.csv
"""
import csv, dataclasses, os, sys
import numpy as np
import jax, jax.numpy as jnp
import optax

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils_cifar import load_cifar10, augment
from pyhgf.model import DeepNetwork
from pyhgf.utils.vectorized_belief_propagation import prediction_pass

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NM_EPOCHS   = 30
SEED        = 0
BASE_LR     = 2.641e-4   # PCX W_LR
WD          = 1.2e-5     # PCX W_WD
TV          = -10.0      # tonic_volatility for all layers

leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)
leaky_relu.__name__ = "leaky_relu"

CONFIGS = [
    # name       padding   use_fc  bs    act_fn            wd     cosine
    ("baseline", "VALID",  True,   32,   leaky_relu,       0.0,   False),
    ("arch",     "SAME",   False,  32,   leaky_relu,       0.0,   False),
    ("bs128",    "VALID",  True,   128,  leaky_relu,       0.0,   False),
    ("adamw",    "VALID",  True,   32,   leaky_relu,       WD,    False),
    ("cosine",   "VALID",  True,   32,   leaky_relu,       0.0,   True),
    ("gelu",     "VALID",  True,   32,   jax.nn.gelu,      0.0,   False),
    ("all",      "SAME",   False,  128,  jax.nn.gelu,      WD,    True),
]


# ── Data ──────────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)
Y_tr = np.eye(10, dtype=np.float32)[y_tr]


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_network(padding, use_fc, act_fn, seed):
    net = DeepNetwork(coupling_fn=act_fn)
    net.add_layer(size=10, kind="binary",
                  tonic_volatility=TV, tonic_volatility_vol=TV,
                  add_constant_input=False, volatility_parent=False)
    if use_fc:
        net.add_layer(size=512,
                      tonic_volatility=TV, tonic_volatility_vol=TV,
                      add_constant_input=True, volatility_parent=False)
    net.add_conv_layer(out_channels=512, kernel_size=3, padding=padding,
                       pool=True, tonic_volatility=TV, tonic_volatility_vol=TV)
    for out_ch in [512, 256, 128]:
        net.add_conv_layer(out_channels=out_ch, kernel_size=3, pool=True,
                           tonic_volatility=TV, tonic_volatility_vol=TV)
    net.add_spatial_input(C=3, H=32, W=32)
    net.weight_initialisation("he", key=jax.random.key(seed))
    return net


def apply_weight_decay(net, wd):
    """Post-epoch L2 weight decay — approximates AdamW for the sweep."""
    if wd <= 0.0:
        return
    new_elements = []
    for elem in net.state.layers:
        w = elem.weights_in
        if w is None:
            new_elements.append(elem)
        elif isinstance(w, tuple):      # conv: (kernel, bias) — decay kernel only
            new_elements.append(
                dataclasses.replace(elem, weights_in=(w[0] * (1.0 - wd), w[1]))
            )
        else:                           # FC / binary weight matrix
            new_elements.append(dataclasses.replace(elem, weights_in=w * (1.0 - wd)))
    net.state = dataclasses.replace(net.state, layers=tuple(new_elements))


def make_cosine_lr(total_steps):
    """Returns an optax schedule callable: step → lr."""
    return optax.warmup_cosine_decay_schedule(
        init_value=BASE_LR,
        peak_value=1.1 * BASE_LR,
        warmup_steps=int(0.1 * total_steps),
        decay_steps=total_steps,
        end_value=0.1 * BASE_LR,
        exponent=1.0,
    )


_X_te_jax = jnp.array(X_te)
# prediction_pass is a top-level @eqx.filter_jit function keyed on Network's
# PyTree structure, so this single jit(vmap(...)) naturally compiles once per
# distinct architecture across the sweep's configs and reuses it every epoch —
# same intent as the old per-instance-id cache, without needing one.
_eval_jit = jax.jit(jax.vmap(prediction_pass, in_axes=(None, 0)))

def evaluate(net):
    preds = np.array(_eval_jit(net.state, _X_te_jax))
    return 100.0 * (np.argmax(preds, axis=1) == y_te).mean()


# ── Sweep ─────────────────────────────────────────────────────────────────────
# NOTE: pyhgf 0.3.0's fit()/propagation_step has no batch-accumulation
# mechanism at all (confirmed: one optax update per sample, unconditionally),
# so `bs` here no longer changes training behavior — the "bs128" config is now
# identical to "baseline" except for its (unused) warm-up slice size. Decide
# whether to fix (remove the bs128 config, since it no longer tests anything)
# or delete before relying on this sweep's results.
os.makedirs(RESULTS_DIR, exist_ok=True)

for (name, padding, use_fc, bs, act_fn, wd, cosine) in CONFIGS:
    # Cosine schedule step count is now per-*sample* (fit() applies one optax
    # update per training sample, not per batch), so total_steps counts total
    # samples across all epochs rather than the old batch-count.
    total_steps = len(X_tr) * NM_EPOCHS
    schedule = make_cosine_lr(total_steps) if cosine else None
    # Passing the schedule directly to optax.adam (rather than rebuilding a
    # fresh optax.adam(lr) every epoch) lets it track the current step via
    # opt_state internally — fit() reinits opt_state whenever the optimizer
    # *object* changes identity, which would otherwise silently reset Adam's
    # momentum every epoch under a changing LR.
    optimizer = optax.adam(schedule if cosine else BASE_LR)

    print(f"\n{'='*60}", flush=True)
    print(f"Config: {name}  padding={padding}  use_fc={use_fc}  bs={bs} (UNUSED, "
          f"no accumulation mechanism in 0.3.0)", flush=True)
    print(f"        act={act_fn.__name__}  wd={wd}  cosine={cosine}", flush=True)

    # JIT warm-up (primes compilation for this architecture)
    print("JIT warm-up...", flush=True)
    _tmp = build_network(padding, use_fc, act_fn, seed=SEED)
    _tmp.fit(X_tr[:bs], Y_tr[:bs], optimizer=optimizer, learning_kind="standard")
    del _tmp
    print("Done.", flush=True)

    # Fresh network for actual training
    net = build_network(padding, use_fc, act_fn, seed=SEED)
    rng = np.random.default_rng(SEED)

    results_path = os.path.join(RESULTS_DIR, f"sweep_hgf_gap_{name}.csv")
    with open(results_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch", "test_acc"]).writeheader()

    for epoch in range(1, NM_EPOCHS + 1):
        idx = rng.permutation(len(X_tr))
        net.fit(augment(X_tr[idx], rng), Y_tr[idx],
                optimizer=optimizer, learning_kind="standard")
        apply_weight_decay(net, wd)

        # Display-only: current LR from the schedule/opt_state's step count.
        lr_display = (
            float(schedule((epoch - 1) * len(X_tr))) if schedule is not None
            else BASE_LR
        )
        acc = evaluate(net)
        print(f"  Epoch {epoch:>2}/{NM_EPOCHS}  lr={lr_display:.2e}  acc={acc:.2f}%",
              flush=True)
        with open(results_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "test_acc"]).writerow(
                {"epoch": epoch, "test_acc": round(acc, 4)})

    print(f"Done → {results_path}", flush=True)

print("\nAll configs complete.", flush=True)
