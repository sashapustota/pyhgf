"""PCX VGG5 on CIFAR-10 — reference for Conv-HGF comparison (~89%).

Matches paper config: T=8, batch_size=128, AdamW+cosine schedule for weights,
SGD+momentum for hidden states, random flip+crop augmentation, 50 epochs.
Results → conv/results/pcx_vgg5_cifar.csv
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

from data_utils_cifar import load_cifar10, augment

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.utils as pxu
import pcx.functional as pxf

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "pcx_vgg5_cifar_hardtanh.csv")

BATCH_SIZE = 128
NM_EPOCHS  = 50
T          = 8
SEED       = 0

# Hyperparameters from VGG5_PCN_CE.yaml
W_LR  = 0.0002641020393177285
W_WD  = 1.2107509557065504e-05
X_LR  = 0.014708110891388754
X_MOM = 0.05

# ── Model ─────────────────────────────────────────────────────────────────────
class VGG5(pxc.EnergyModule):
    def __init__(self, nm_classes: int, act_fn: Callable) -> None:
        super().__init__()
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (pxnn.Conv2d(3, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(128, 256, kernel_size=(3,3), padding=(1,1)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(256, 512, kernel_size=(3,3), padding=(1,1)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            # Paper Table: last conv has padding=0 → 4×4 → 2×2 → pool → 1×1
            (pxnn.Conv2d(512, 512, kernel_size=(3,3), padding=(0,0)),
             self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
        ]
        self.classifier_layers = [
            (pxnn.Linear(512 * 1 * 1, self.nm_classes.get()),),
        ]
        self.vodes = (
            [pxc.Vode() for _ in self.feature_layers] +
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


# ── Training functions ────────────────────────────────────────────────────────
@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
          in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: VGG5):
    return model(x, y)

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
          in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: VGG5):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_

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
    optim_w.step(model, g["model"])

@pxf.jit()
def eval_on_batch(x, y, *, model):
    model.eval()
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)
    return (y_ == y).mean(), y_

# ── Data ──────────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)

steps_per_epoch = len(X_tr) // BATCH_SIZE
total_steps     = steps_per_epoch * NM_EPOCHS

# ── Model + optimisers ────────────────────────────────────────────────────────
model = VGG5(nm_classes=10, act_fn=jax.nn.hard_tanh)

schedule = optax.warmup_cosine_decay_schedule(
    init_value=W_LR,
    peak_value=1.1 * W_LR,
    warmup_steps=int(0.1 * total_steps),
    decay_steps=total_steps,
    end_value=0.1 * W_LR,
    exponent=1.0)

with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(jnp.zeros((BATCH_SIZE, 3, 32, 32)), None, model=model)
    optim_h = pxu.Optim(lambda: optax.sgd(X_LR, momentum=X_MOM),
                        pxu.M(pxc.VodeParam)(model))
    optim_w = pxu.Optim(lambda: optax.adamw(schedule, weight_decay=W_WD),
                        pxu.M(pxnn.LayerParam)(model))

print(f"T={T}  batch_size={BATCH_SIZE}  epochs={NM_EPOCHS}", flush=True)
print(f"W_LR={W_LR:.2e}  X_LR={X_LR:.2e}  X_MOM={X_MOM}\n", flush=True)

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

rng = np.random.default_rng(SEED)

for epoch in range(1, NM_EPOCHS + 1):
    perm = rng.permutation(len(X_tr))
    for i in range(0, len(X_tr) - BATCH_SIZE + 1, BATCH_SIZE):
        idx     = perm[i:i + BATCH_SIZE]
        x_batch = augment(X_tr[idx], rng)
        y_batch = jax.nn.one_hot(y_tr[idx], 10)
        train_on_batch(T, x_batch, y_batch,
                       model=model, optim_w=optim_w, optim_h=optim_h)

    accs = []
    for i in range(0, len(X_te) - BATCH_SIZE + 1, BATCH_SIZE):
        a, _ = eval_on_batch(X_te[i:i + BATCH_SIZE], y_te[i:i + BATCH_SIZE], model=model)
        accs.append(float(a))
    acc = 100.0 * float(np.mean(accs))
    print(f"Epoch {epoch:>2}/{NM_EPOCHS}  acc={acc:.2f}%", flush=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {"epoch": epoch, "test_acc": round(acc, 4)})

print(f"\nDone. Results saved to {RESULTS_PATH}", flush=True)
