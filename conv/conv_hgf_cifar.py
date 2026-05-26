"""Phase 2: Conv-HGF on CIFAR-10.

Architecture (VGG5-like, 4 conv blocks + optional FC head):
  Input: (3, 32, 32)
  Conv1: 64  ch, 3×3, same → pool 2×2 → (64, 16, 16)
  Conv2: 128 ch, 3×3, same → pool 2×2 → (128, 8, 8)
  Conv3: 256 ch, 3×3, same → pool 2×2 → (256, 4, 4)
  Conv4: 512 ch, 3×3, same → pool 2×2 → (512, 2, 2)
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "conv_hgf_cifar_{}.csv".format("fc" if os.environ.get("USE_FC", "1") != "0" else "nofc"))

TONIC_VOL     = -10.0
TONIC_VOL_VOL = -10.0
BASE_LR       = 0.0001
BATCH_SIZE    = 32
LR            = BASE_LR * BATCH_SIZE   # linear scaling rule: lr ∝ batch_size
LEARNING_KIND = "standard"
EPOCHS        = 50
SEED          = 0
USE_FC        = os.environ.get("USE_FC", "1") != "0"  # override: USE_FC=0 python ...

leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)

# ── Data ──────────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)
# X_tr: (50000, 3, 32, 32) — keep as spatial, no flattening
Y_tr = np.eye(10, dtype=np.float32)[y_tr]

# ── Network ───────────────────────────────────────────────────────────────────
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
    for out_ch in [512, 256, 128, 64]:
        net.add_conv_layer(out_channels=out_ch,
                           kernel_size=3,
                           pool=True,
                           tonic_volatility=TONIC_VOL,
                           tonic_volatility_vol=TONIC_VOL_VOL)
    net.add_spatial_input(C=3, H=32, W=32)
    net.weight_initialisation(strategy="he", seed=seed)
    return net

def evaluate(net):
    preds = np.array(net.predict(jnp.array(X_te)))
    return 100.0 * (np.argmax(preds, axis=1) == y_te).mean()

# ── JIT warm-up ───────────────────────────────────────────────────────────────
print(f"LR={LR}  (base={BASE_LR} × batch={BATCH_SIZE})  use_fc={USE_FC}", flush=True)
print("JIT warm-up...", flush=True)
_net = build_network(seed=0)
_net.fit(X_tr[:4], Y_tr[:4], lr=LR, learning_kind=LEARNING_KIND, batch_size=BATCH_SIZE)
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
    net.fit(X_aug, Y_tr[idx], lr=LR, learning_kind=LEARNING_KIND, batch_size=BATCH_SIZE)
    acc = evaluate(net)
    print(f"Epoch {epoch:>2}/{EPOCHS}  acc={acc:.2f}%", flush=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {"epoch": epoch, "test_acc": round(acc, 4)})

print(f"\nDone. Results saved to {RESULTS_PATH}", flush=True)
