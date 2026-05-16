"""PyTorch CNN baseline sweep for CIFAR-10.

Matches sweep_cifar.py configs:
  - Same architecture (VGG5-like, 4 conv + FC head)
  - Same LRs, epochs, training samples, seed
Grid: 3 learning rates, Adam optimizer.
Results → conv/results/sweep_cifar_pytorch.csv
"""
import csv, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils_cifar import load_cifar10

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "sweep_cifar_pytorch.csv")

EPOCHS   = 10
N_TRAIN  = 20_000
SEED     = 0
BATCH    = 64

LRS = [0.001, 0.0005, 0.0001]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data ───────────────────────────────────────────────────────────────────────
X_full, y_full, X_te, y_te = load_cifar10(DATA_DIR)
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X_full))[:N_TRAIN]
X_tr = X_full[idx]
y_tr = y_full[idx]

X_tr_t = torch.tensor(X_tr).float()
y_tr_t = torch.tensor(y_tr).long()
X_te_t = torch.tensor(X_te).float().to(DEVICE)
y_te_t = torch.tensor(y_te).long().to(DEVICE)

# ── Model ──────────────────────────────────────────────────────────────────────
class VGG5(nn.Module):
    """4-conv VGG-like net matching Conv-HGF architecture."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model():
    model = VGG5().to(DEVICE)
    torch.nn.init.kaiming_normal_(model.head[1].weight)
    torch.nn.init.kaiming_normal_(model.head[3].weight)
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
    return model

def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        preds = logits.argmax(dim=1)
        acc = 100.0 * (preds == y_te_t).float().mean().item()
    model.train()
    return acc

# ── Sweep ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["lr", "epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

print(f"Sweeping {len(LRS)} LRs × {EPOCHS} epochs × {N_TRAIN} samples\n")

best = {}

for combo_i, lr in enumerate(LRS, 1):
    print(f"[{combo_i}/{len(LRS)}]  lr={lr}", flush=True)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    sweep_rng = np.random.default_rng(SEED)
    combo_best = 0.0

    for epoch in range(1, EPOCHS + 1):
        perm = sweep_rng.permutation(N_TRAIN)
        ds = TensorDataset(X_tr_t[perm], y_tr_t[perm])
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        acc = evaluate(model)
        combo_best = max(combo_best, acc)
        print(f"  epoch {epoch:>2}/{EPOCHS}  acc={acc:5.2f}%", flush=True)
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
                {"lr": lr, "epoch": epoch, "test_acc": round(acc, 4)})

    best[lr] = combo_best
    print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("── Summary (best test acc per LR) ──")
for lr, acc in sorted(best.items(), key=lambda x: -x[1]):
    print(f"  lr={lr:<7}  best={acc:.2f}%")
print(f"\nResults saved to {RESULTS_PATH}")
