"""Backprop VGG5 on CIFAR-10 — baseline comparison for PCX VGG5 (~88%).

Same architecture as pcx_vgg5_cifar.py: paddings [1,1,1,0] per paper Table,
same augmentation, same 50 epochs / batch 128.
Paper reports BP-CE = 88.11 ± 0.13%.
Results → conv/results/bp_vgg5_cifar.csv
"""
import csv, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar10")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils_cifar import load_cifar10, augment

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "bp_vgg5_cifar.csv")

BATCH_SIZE = 128
NM_EPOCHS  = 50
SEED       = 0
LR         = 0.1
MOMENTUM   = 0.9
WEIGHT_DECAY = 5e-4

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


class VGG5(nn.Module):
    def __init__(self, nm_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(3,   128, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 16×16 → 8×8
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 8×8 → 4×4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 4×4 → 2×2 (padding=0) → 1×1
            nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(512, nm_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ── Data ──────────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)

# Convert to torch tensors once; augmentation is applied per-batch on numpy
X_te_t = torch.from_numpy(X_te).to(device)
y_te_t  = torch.from_numpy(y_te).to(device)

# ── Model + optimiser ─────────────────────────────────────────────────────────
model = VGG5().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}", flush=True)

optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDNAMES = ["epoch", "test_acc"]
with open(RESULTS_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

rng = np.random.default_rng(SEED)

print(f"LR={LR}  momentum={MOMENTUM}  wd={WEIGHT_DECAY}  "
      f"epochs={NM_EPOCHS}  batch={BATCH_SIZE}\n", flush=True)

for epoch in range(1, NM_EPOCHS + 1):
    model.train()
    perm = rng.permutation(len(X_tr))
    X_aug = augment(X_tr[perm], rng)

    for i in range(0, len(X_aug) - BATCH_SIZE + 1, BATCH_SIZE):
        x = torch.from_numpy(X_aug[i:i + BATCH_SIZE]).to(device)
        y = torch.from_numpy(y_tr[perm[i:i + BATCH_SIZE]]).to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        accs = []
        for i in range(0, len(X_te_t) - BATCH_SIZE + 1, BATCH_SIZE):
            logits = model(X_te_t[i:i + BATCH_SIZE])
            acc = (logits.argmax(1) == y_te_t[i:i + BATCH_SIZE]).float().mean()
            accs.append(acc.item())
    acc = 100.0 * float(np.mean(accs))
    print(f"Epoch {epoch:>2}/{NM_EPOCHS}  acc={acc:.2f}%  "
          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
            {"epoch": epoch, "test_acc": round(acc, 4)})

print(f"\nDone. Results saved to {RESULTS_PATH}", flush=True)
