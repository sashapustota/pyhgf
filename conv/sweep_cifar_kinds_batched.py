"""Learning-kind sweep for Conv-HGF on CIFAR-10 with mini-batch gradient accumulation.

Mirrors sweep_cifar_kinds.py but applies the linear LR scaling rule:
  lr = BASE_LR * BATCH_SIZE
Results → conv/results/sweep_cifar_kinds_batched.csv
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "sweep_cifar_kinds_batched.csv")

TONIC_VOL  = -10.0
BASE_LR    = 0.0001
BATCH_SIZE = 32
# pyhgf 0.3.0's fit()/propagation_step has no batch-accumulation mechanism at
# all (confirmed: one optax update per sample, unconditionally) — this script's
# whole premise (mini-batch gradient accumulation) no longer exists on this
# architecture. LR is left at the batch-scaled value from the old regime
# rather than silently rescaling it down, since that would make this script a
# silent duplicate of sweep_cifar_kinds.py. As-is this LR is ~32x too large
# for true per-sample updates and will likely diverge — decide whether to fix
# (rescale LR, rename to reflect it's no longer batched) or delete this
# script before running it for real.
LR         = BASE_LR * BATCH_SIZE   # linear scaling rule (STALE — see above)
EPOCHS     = 10
N_TRAIN    = 20_000
SEED       = 0

KINDS = ["standard", "precision_weighted", "precision_ratio"]

leaky_relu = lambda x: jnp.where(x > 0, x, 0.01 * x)

# ── Data ───────────────────────────────────────────────────────────────────────
X_full, y_full, X_te, y_te = load_cifar10(DATA_DIR)
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X_full))[:N_TRAIN]
X_tr = X_full[idx]
Y_tr = np.eye(10, dtype=np.float32)[y_full[idx]]

# ── Network builder ────────────────────────────────────────────────────────────
def build_network(seed):
    net = DeepNetwork(coupling_fn=leaky_relu)
    net.add_layer(size=10, kind="binary",
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=False, volatility_parent=False)
    net.add_layer(size=512,
                  tonic_volatility=TONIC_VOL, tonic_volatility_vol=TONIC_VOL,
                  add_constant_input=True, volatility_parent=False)
    for out_ch in [512, 256, 128, 64]:
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
print(f"batch_size={BATCH_SIZE} (UNUSED, no accumulation mechanism in 0.3.0)  "
      f"lr={LR}  (base={BASE_LR} × {BATCH_SIZE})", flush=True)
print("JIT warm-up...", flush=True)
for kind in KINDS:
    _net = build_network(0)
    _net.fit(X_tr[:BATCH_SIZE], Y_tr[:BATCH_SIZE], optimizer=OPTIMIZER,
             learning_kind=kind)
    print(f"  warmed up: {kind}", flush=True)
print("Done.\n", flush=True)

# ── Sweep ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["kind", "epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

print(f"epochs={EPOCHS}  n_train={N_TRAIN}\n")

best = {}
for i, kind in enumerate(KINDS, 1):
    print(f"[{i}/{len(KINDS)}]  kind={kind}", flush=True)
    net = build_network(SEED)
    sweep_rng = np.random.default_rng(SEED)
    combo_best = 0.0
    for epoch in range(1, EPOCHS + 1):
        perm = sweep_rng.permutation(N_TRAIN)
        net.fit(X_tr[perm], Y_tr[perm], optimizer=OPTIMIZER, learning_kind=kind)
        acc = evaluate(net)
        combo_best = max(combo_best, acc)
        print(f"  epoch {epoch}/{EPOCHS}  acc={acc:5.2f}%", flush=True)
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
                {"kind": kind, "epoch": epoch, "test_acc": round(acc, 4)})
    best[kind] = combo_best
    print()

print("── Summary ──")
for kind, acc in sorted(best.items(), key=lambda x: -x[1]):
    print(f"  {kind:<22}  best={acc:.2f}%")
print(f"\nResults saved to {RESULTS_PATH}")
