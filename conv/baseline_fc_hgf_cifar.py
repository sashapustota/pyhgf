"""Phase 1: FC-HGF baseline on CIFAR-10.

Architecture (mirrors Kaggle MLP achieving ~52% with backprop):
  3072 → 512 → 256 → 128 → 10

Output: conv/results/baseline_fc_hgf_cifar.csv
"""
import csv, os, sys
import numpy as np
import jax, jax.numpy as jnp
import optax

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils_cifar import load_cifar10
from pyhgf.model import DeepNetwork

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "baseline_fc_hgf_cifar.csv")

OUTPUT_DIM    = 10
TONIC_VOL     = -10.0
TONIC_VOL_VOL = -4.0
LR            = 0.001
EPOCHS        = 50
SEED          = 0

leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)
linear     = lambda x: x

# ── Data ──────────────────────────────────────────────────────────────────────
X_tr_4d, y_tr, X_te_4d, y_te = load_cifar10(DATA_DIR)
X_tr = X_tr_4d.reshape(len(X_tr_4d), -1).astype(np.float32)
X_te = X_te_4d.reshape(len(X_te_4d), -1).astype(np.float32)
Y_tr = np.eye(OUTPUT_DIM, dtype=np.float32)[y_tr]

# ── Network ───────────────────────────────────────────────────────────────────
def build_network(seed):
    net = DeepNetwork(coupling_fn=leaky_relu)
    net.add_layer(size=OUTPUT_DIM, kind="binary",
                  tonic_volatility=TONIC_VOL,
                  tonic_volatility_vol=TONIC_VOL_VOL,
                  add_constant_input=False)
    for i, width in enumerate([128, 256, 512]):
        net.add_layer(size=width,
                      tonic_volatility=TONIC_VOL,
                      tonic_volatility_vol=TONIC_VOL_VOL,
                      add_constant_input=(i > 0),
                      volatility_parent=False)
    net.add_layer(size=3072,
                  tonic_volatility=TONIC_VOL,
                  tonic_volatility_vol=TONIC_VOL_VOL,
                  add_constant_input=False,
                  coupling_fn=linear,
                  volatility_parent=False)
    net.weight_initialisation("he", key=jax.random.key(seed))
    return net

def evaluate(net):
    preds = np.array(net.predict(jnp.array(X_te)))
    return 100.0 * (np.argmax(preds, axis=1) == y_te).mean()

# fit() compares the optimizer by identity to decide whether to reinit
# opt_state, so the same instance must be reused across every fit() call.
OPTIMIZER = optax.sgd(LR)

# ── JIT warm-up ───────────────────────────────────────────────────────────────
print("JIT warm-up...", flush=True)
_net = build_network(seed=0)
_net.fit(X_tr[:64], Y_tr[:64], optimizer=OPTIMIZER, learning_kind="precision_weighted")
print("Done.\n", flush=True)

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

rng = np.random.default_rng(SEED)
net = build_network(SEED)

for epoch in range(1, EPOCHS + 1):
    idx = rng.permutation(len(X_tr))
    net.fit(X_tr[idx], Y_tr[idx], optimizer=OPTIMIZER, learning_kind="precision_weighted")
    acc = evaluate(net)
    print(f"Epoch {epoch:>2}/{EPOCHS}  acc={acc:.2f}%", flush=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {"epoch": epoch, "test_acc": round(acc, 4)})

print(f"\nDone. Results saved to {RESULTS_PATH}", flush=True)
