"""Phase 2: Conv-HGF on CIFAR-10.

Architecture (VGG5, matching PCX channel widths and spatial output):
  Input: (3, 32, 32)
  Conv1: 128 ch, 3×3, SAME  → pool 2×2 → (128, 16, 16)
  Conv2: 256 ch, 3×3, SAME  → pool 2×2 → (256,  8,  8)
  Conv3: 512 ch, 3×3, SAME  → pool 2×2 → (512,  4,  4)
  Conv4: 512 ch, 3×3, VALID → pool 2×2 → (512,  1,  1)
  FC:    512 nodes  (only when USE_FC=True)
  Out:   10 binary nodes

Output: conv/results/conv_hgf_cifar.csv
"""
import csv, os, sys
import numpy as np
import jax, jax.numpy as jnp

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils_cifar import load_cifar10, augment
from pyhgf.model import DeepNetwork

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_tag = ("fc" if os.environ.get("USE_FC", "1") != "0" else "nofc") + "_adam_gelu"
RESULTS_PATH = os.path.join(RESULTS_DIR, f"conv_hgf_cifar_{_tag}.csv")

TONIC_VOL     = -10.0
TONIC_VOL_VOL = -10.0
ADAM_LR       = 2.641e-4   # matches PCX W_LR from VGG5_PCN_CE.yaml
BATCH_SIZE    = 32
LEARNING_KIND = "standard"
EPOCHS        = 50
SEED          = 0
USE_FC        = os.environ.get("USE_FC", "1") != "0"  # override: USE_FC=0 python ...

# ── Data ──────────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)
# X_tr: (50000, 3, 32, 32) — keep as spatial, no flattening
Y_tr = np.eye(10, dtype=np.float32)[y_tr]

# ── Network ───────────────────────────────────────────────────────────────────
def build_network(seed):
    net = DeepNetwork(coupling_fn=jax.nn.gelu)
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
    # Last conv (closest to FC head): VALID padding → 4×4 → 2×2 → pool → 1×1
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
    net.weight_initialisation(strategy="he", seed=seed)
    return net

_X_te_jax = jnp.array(X_te)
_eval_jit  = {}

def evaluate(net):
    key = id(net)
    if key not in _eval_jit:
        if net._prediction_fn is None:
            net._prediction_fn = net._create_prediction_fn()
        pf = net._prediction_fn
        _eval_jit[key] = jax.jit(
            lambda state, x: jax.vmap(lambda xi: pf(state, xi))(x)
        )
    preds = np.array(_eval_jit[key](net.state, _X_te_jax))
    return 100.0 * (np.argmax(preds, axis=1) == y_te).mean()

ADAM_PARAMS = {"lr": ADAM_LR}

# ── JIT warm-up ───────────────────────────────────────────────────────────────
print(f"Adam LR={ADAM_LR}  batch={BATCH_SIZE}  use_fc={USE_FC}", flush=True)
print("JIT warm-up...", flush=True)
_net = build_network(seed=0)
_net.fit(X_tr[:4], Y_tr[:4], lr="adam", learning_kind=LEARNING_KIND,
         params=ADAM_PARAMS, batch_size=BATCH_SIZE)
print("Done.\n", flush=True)

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

rng = np.random.default_rng(SEED)
net = build_network(SEED)

acc0 = evaluate(net)
print(f"Epoch  0/{EPOCHS}  acc={acc0:.2f}%  (init, should be ~10%)", flush=True)
with open(RESULTS_PATH, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
        {"epoch": 0, "test_acc": round(acc0, 4)})

for epoch in range(1, EPOCHS + 1):
    idx = rng.permutation(len(X_tr))
    X_aug = augment(X_tr[idx], rng)
    net.fit(X_aug, Y_tr[idx], lr="adam", learning_kind=LEARNING_KIND,
            params=ADAM_PARAMS, batch_size=BATCH_SIZE)
    acc = evaluate(net)
    print(f"Epoch {epoch:>2}/{EPOCHS}  acc={acc:.2f}%", flush=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {"epoch": epoch, "test_acc": round(acc, 4)})

print(f"\nDone. Results saved to {RESULTS_PATH}", flush=True)
