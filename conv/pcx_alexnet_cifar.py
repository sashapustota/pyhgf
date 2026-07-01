"""PCX AlexNet on CIFAR-10 — reference comparison for Conv-HGF.

Architecture: AlexNet (5 conv + 2 FC + output), T=13 inference steps/batch.
batch_size=128, AdamW (weights) + SGD momentum (hidden states), 24 epochs.
Results → conv/results/pcx_alexnet_cifar.csv
"""
import csv, os, sys
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import optax

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pcx"))

from data_utils_cifar import load_cifar10

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "pcx_alexnet_cifar.csv")

BATCH_SIZE = 128
NM_EPOCHS  = 24
T          = 13
SEED       = 0

# ── Model ─────────────────────────────────────────────────────────────────────
class AlexNet(pxc.EnergyModule):
    def __init__(self, nm_classes: int, act_fn: Callable) -> None:
        super().__init__()
        self.nm_classes = nm_classes
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (pxnn.Conv2d(3, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(64, 192, kernel_size=(3,3), padding=(1,1)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)), self.act_fn),
            (pxnn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)), self.act_fn),
            (pxnn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
        ]
        self.classifier_layers = [
            (pxnn.Linear(256 * 2 * 2, 4096), self.act_fn),
            (pxnn.Linear(4096, 4096),        self.act_fn),
            (pxnn.Linear(4096, nm_classes),),
        ]
        self.vodes = (
            [pxc.Vode() for _ in self.feature_layers] +
            [pxc.Vode() for _ in self.classifier_layers[:-1]] +
            [pxc.Vode(pxc.ce_energy)]
        )
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array):
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x)
        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x)
        if y is not None:
            self.vodes[-1].set("h", y)
        return self.vodes[-1].get("u")


# ── Training functions (identical to PCX notebook) ────────────────────────────
@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
          in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: AlexNet):
    return model(x, y)

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
          in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: AlexNet):
    y_ = model(x, None)
    return jax.lax.psum(model.energy(), "batch"), y_

@pxf.jit(static_argnums=0)
def train_on_batch(T, x, y, *, model, optim_w, optim_h):
    model.train()
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True)(energy)(x, model=model)
        optim_h.step(model, g["model"])
    optim_h.clear()
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(
            pxu.M(pxnn.LayerParam).to([False, True]),
            has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0])

@pxf.jit()
def eval_on_batch(x, y, *, model):
    model.eval()
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)
    return (y_ == y).mean(), y_

# ── Data ──────────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)
Y_tr = jax.nn.one_hot(y_tr, 10)

# ── Model + optimisers ────────────────────────────────────────────────────────
model   = AlexNet(nm_classes=10, act_fn=jax.nn.gelu)
optim_h = pxu.Optim(lambda: optax.sgd(1e-2, momentum=0.5, nesterov=True))
optim_w = pxu.Optim(lambda: optax.adamw(1e-4), pxu.M(pxnn.LayerParam)(model))

print(f"batch_size={BATCH_SIZE}  T={T}  epochs={NM_EPOCHS}", flush=True)
print("JIT warm-up...", flush=True)
train_on_batch(T, X_tr[:BATCH_SIZE], Y_tr[:BATCH_SIZE],
               model=model, optim_w=optim_w, optim_h=optim_h)
print("Done.\n", flush=True)

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

rng = np.random.default_rng(SEED)

for epoch in range(1, NM_EPOCHS + 1):
    perm = rng.permutation(len(X_tr))
    for i in range(0, len(X_tr) - BATCH_SIZE + 1, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        train_on_batch(T, X_tr[idx], Y_tr[idx],
                       model=model, optim_w=optim_w, optim_h=optim_h)

    # Evaluate
    accs = []
    for i in range(0, len(X_te) - BATCH_SIZE + 1, BATCH_SIZE):
        a, _ = eval_on_batch(X_te[i:i+BATCH_SIZE], y_te[i:i+BATCH_SIZE], model=model)
        accs.append(float(a))
    acc = 100.0 * float(np.mean(accs))
    print(f"Epoch {epoch:>2}/{NM_EPOCHS}  acc={acc:.2f}%", flush=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {"epoch": epoch, "test_acc": round(acc, 4)})

print(f"\nDone. Results saved to {RESULTS_PATH}", flush=True)
