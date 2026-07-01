"""Architecture sweep for Conv-HGF on CIFAR-10.

Grid: 2 widths × 3 depths = 6 architectures.
Fixed: lr=0.0001, kind=standard, 10 epochs, 20k samples.
Results → conv/results/sweep_cifar_arch.csv
"""
import csv, os, sys, itertools
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "sweep_cifar_arch.csv")

TONIC_VOL = -10.0
LR        = 0.0001
KIND      = "standard"
EPOCHS    = 10
N_TRAIN   = 20_000
SEED      = 0

leaky_relu = lambda x: jnp.where(x > 0, x, 0.01 * x)

# width name → conv channel sequence (output → input order, as added to DeepNetwork)
WIDTHS = {
    "narrow":  [256, 128, 64, 32],
    "current": [512, 256, 128, 64],
}
DEPTHS = [2, 3, 4]

# ── Data ───────────────────────────────────────────────────────────────────────
X_full, y_full, X_te, y_te = load_cifar10(DATA_DIR)
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X_full))[:N_TRAIN]
X_tr = X_full[idx]
Y_tr = np.eye(10, dtype=np.float32)[y_full[idx]]

# ── Network builder ────────────────────────────────────────────────────────────
def build_network(depth, channels, seed):
    """Build a Conv-HGF network with given depth and channel sequence."""
    net = DeepNetwork(coupling_fn=leaky_relu)
    net.add_layer(size=10, kind="binary",
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=False)
    fc_size = channels[-depth]  # innermost conv channels = FC input width
    net.add_layer(size=fc_size,
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=True, volatility_parent=False)
    for out_ch in channels[-depth:]:
        net.add_conv_layer(out_channels=out_ch, kernel_size=3, pool=True,
                           tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL)
    net.add_spatial_input(C=3, H=32, W=32)
    net.weight_initialisation("he", key=jax.random.key(seed))
    return net

def evaluate(net):
    preds = np.array(net.predict(jnp.array(X_te)))
    return 100.0 * (np.argmax(preds, axis=1) == y_te).mean()

# fit() compares the optimizer by identity to decide whether to reinit
# opt_state, so the same instance must be reused across every fit() call.
OPTIMIZER = optax.sgd(LR)

# ── JIT warm-up ────────────────────────────────────────────────────────────────
print("JIT warm-up...", flush=True)
for width_name, channels in WIDTHS.items():
    for depth in DEPTHS:
        _net = build_network(depth, channels, 0)
        _net.fit(X_tr[:4], Y_tr[:4], optimizer=OPTIMIZER, learning_kind=KIND)
        print(f"  warmed up: depth={depth}  width={width_name}", flush=True)
print("Done.\n", flush=True)

# ── Sweep ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["width", "depth", "epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

combos = list(itertools.product(WIDTHS.items(), DEPTHS))
print(f"Sweeping {len(combos)} combos × {EPOCHS} epochs × {N_TRAIN} samples\n")

best = {}

for combo_i, ((width_name, channels), depth) in enumerate(combos, 1):
    print(f"[{combo_i}/{len(combos)}]  width={width_name}  depth={depth}", flush=True)
    net = build_network(depth, channels, SEED)
    sweep_rng = np.random.default_rng(SEED)
    combo_best = 0.0
    for epoch in range(1, EPOCHS + 1):
        perm = sweep_rng.permutation(N_TRAIN)
        net.fit(X_tr[perm], Y_tr[perm], optimizer=OPTIMIZER, learning_kind=KIND)
        acc = evaluate(net)
        combo_best = max(combo_best, acc)
        print(f"  epoch {epoch:>2}/{EPOCHS}  acc={acc:5.2f}%", flush=True)
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
                {"width": width_name, "depth": depth, "epoch": epoch,
                 "test_acc": round(acc, 4)})
    best[(width_name, depth)] = combo_best
    print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("── Summary (best test acc per architecture) ──")
for (width_name, depth), acc in sorted(best.items(), key=lambda x: -x[1]):
    print(f"  width={width_name:<8}  depth={depth}  best={acc:.2f}%")
print(f"\nResults saved to {RESULTS_PATH}")
