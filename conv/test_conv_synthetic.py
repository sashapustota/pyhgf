"""Synthetic data smoke-test for Conv-HGF.

Two architectures tested back-to-back:

  2-conv (16×16 input):
    Input: (1, 16, 16)
    Conv1: 16 ch, 3×3 → pool 2×2 → (16, 8, 8)
    Conv2: 32 ch, 3×3 → pool 2×2 → (32, 4, 4)
    FC: 64  →  Out: 4 binary

  4-conv (32×32 input, same data zero-padded):
    Input: (1, 32, 32)
    Conv1: 8  ch, 3×3 → pool 2×2 → (8,  16, 16)
    Conv2: 16 ch, 3×3 → pool 2×2 → (16,  8,  8)
    Conv3: 32 ch, 3×3 → pool 2×2 → (32,  4,  4)
    Conv4: 64 ch, 3×3 → pool 2×2 → (64,  2,  2)
    FC: 64  →  Out: 4 binary

Signal: faint vertical edge (±0.5) in one quadrant per class (SNR ~0.5).
Chance = 25%.  If 4-conv fails while 2-conv passes, gradient depth is the issue.
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
import optax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyhgf.model import DeepNetwork

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_TRAIN   = 400
N_TEST    = 100
TONIC_VOL = -10.0   # less aggressive than -10; change this to test sensitivity
LR        = 0.001
EPOCHS    = 20
SEED      = 0

leaky_relu = lambda x: jnp.where(x > 0, x, 0.01 * x)

# ── Synthetic data ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def make_data(n, rng):
    X = rng.standard_normal((n, 1, 16, 16)).astype(np.float32)
    y = rng.integers(0, 4, size=n)
    # Faint vertical edge in one quadrant per class (SNR ~0.5)
    strength = 0.5
    for cls, (rs, re, cs, ce) in enumerate(
        [(0, 8, 0, 8), (0, 8, 8, 16), (8, 16, 0, 8), (8, 16, 8, 16)]
    ):
        mask = y == cls
        X[mask, 0, rs:re, cs : cs + (ce - cs) // 2] += strength
        X[mask, 0, rs:re, cs + (ce - cs) // 2 : ce] -= strength
    return X, y

X_tr, y_tr = make_data(N_TRAIN, rng)
X_te, y_te = make_data(N_TEST,  rng)
Y_tr = np.eye(4, dtype=np.float32)[y_tr]

# Zero-pad to 32×32 for the 4-conv run
X_tr32 = np.pad(X_tr, ((0, 0), (0, 0), (8, 8), (8, 8)))
X_te32 = np.pad(X_te, ((0, 0), (0, 0), (8, 8), (8, 8)))

# ── Network ────────────────────────────────────────────────────────────────────
def build_2conv(seed):
    net = DeepNetwork(coupling_fn=leaky_relu)
    net.add_layer(size=4, kind="binary",
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=False, volatility_parent=False)
    net.add_layer(size=64,
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=True, volatility_parent=False)
    for out_ch in [32, 16]:
        net.add_conv_layer(out_channels=out_ch, kernel_size=3, pool=True,
                           tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL)
    net.add_spatial_input(C=1, H=16, W=16)
    net.weight_initialisation("he", key=jax.random.key(seed))
    return net

def build_4conv(seed):
    net = DeepNetwork(coupling_fn=leaky_relu)
    net.add_layer(size=4, kind="binary",
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=False, volatility_parent=False)
    net.add_layer(size=64,
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=True, volatility_parent=False)
    for out_ch in [64, 32, 16, 8]:
        net.add_conv_layer(out_channels=out_ch, kernel_size=3, pool=True,
                           tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL)
    net.add_spatial_input(C=1, H=32, W=32)
    net.weight_initialisation("he", key=jax.random.key(seed))
    return net

def evaluate(net, X, y):
    preds = np.array(net.predict(jnp.array(X)))
    return 100.0 * (np.argmax(preds, axis=1) == y).mean()

# fit() compares the optimizer by identity to decide whether to reinit
# opt_state, so the same instance must be reused across every fit() call for
# a given network. Sharing one instance across networks is safe (each
# DeepNetwork tracks its own opt_state independently).
OPTIMIZER = optax.sgd(LR)

# ── JIT warm-up ────────────────────────────────────────────────────────────────
print("JIT warm-up (2-conv)...", flush=True)
_net = build_2conv(seed=0)
_net.fit(X_tr[:2], Y_tr[:2], optimizer=OPTIMIZER, learning_kind="precision_weighted")
print("JIT warm-up (4-conv)...", flush=True)
_net4 = build_4conv(seed=0)
_net4.fit(X_tr32[:2], Y_tr[:2], optimizer=OPTIMIZER, learning_kind="precision_weighted")
print("Done.\n", flush=True)

# ── 2-conv run ─────────────────────────────────────────────────────────────────
print(f"── 2-conv  TONIC_VOL={TONIC_VOL}  LR={LR} ──")
net2 = build_2conv(SEED)
rng2 = np.random.default_rng(SEED)
for epoch in range(1, EPOCHS + 1):
    idx = rng2.permutation(N_TRAIN)
    net2.fit(X_tr[idx], Y_tr[idx], optimizer=OPTIMIZER, learning_kind="precision_weighted")
    acc = evaluate(net2, X_te, y_te)
    print(f"  Epoch {epoch:>2}/{EPOCHS}  acc={acc:5.1f}%  {'#' * int(acc / 5)}", flush=True)

# ── 4-conv run ─────────────────────────────────────────────────────────────────
print(f"\n── 4-conv  TONIC_VOL={TONIC_VOL}  LR={LR} ──")
net4 = build_4conv(SEED)
rng4 = np.random.default_rng(SEED)
for epoch in range(1, EPOCHS + 1):
    idx = rng4.permutation(N_TRAIN)
    net4.fit(X_tr32[idx], Y_tr[idx], optimizer=OPTIMIZER, learning_kind="precision_weighted")
    acc = evaluate(net4, X_te32, y_te)
    print(f"  Epoch {epoch:>2}/{EPOCHS}  acc={acc:5.1f}%  {'#' * int(acc / 5)}", flush=True)

print("\nDone.")
