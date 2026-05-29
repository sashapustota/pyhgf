"""
ViT-B/16 hybrid HGF transformer — online block adaptation (Level 1.5).

Three conditions on frozen ViT-B/16 backbone:
  A. Frozen hybrid     — all 12 HGF MLP blocks static (no adaptation)
  B. HGF-adapt         — block 11 updated online via precision-weighted targets
  C. Adam fine-tune    — block 11 MLP updated via standard backprop (Adam)

Block update mechanism for B (two-pass):
  1. Forward through blocks 0–10 (predict-only HGF).
  2. Block 11: predict delta_11 with JAX, wrap as PyTorch leaf tensor.
  3. Downstream PyTorch: residual + LN + frozen probe → cross-entropy loss.
  4. Backprop → ∂L/∂delta_11 (non-zero only at CLS token position).
  5. Target: target_11 = delta_11 - step_size * grad_11.
  6. HGF update: hgf_blocks[11].fit(h_11, target_11).

Experiment: Split-CIFAR-10, 5 tasks × 2 classes.
Metric: accuracy matrix A[i,j] and BWT.

Caching: blocks 0–10 are frozen across all conditions, so their intermediate
activations (tokens after block-11 attention + LN2 output = block-11 MLP input)
are precomputed once and reused. This eliminates the 11-block forward pass from
the inner training/evaluation loop.

Run with:
    python experiments/vit_hgf_adapt.py [--step-size 1e-3] [--n-eval 400]
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT      = Path(__file__).parent.parent
CACHE_DIR = Path(__file__).parent / "cifar10_vit_embeddings"
DATA_DIR  = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
sys.path.insert(0, str(ROOT))

from pyhgf.model import DeepNetwork
from pyhgf.typing import LayerState

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM  = 768
MLP_DIM    = 3072
N_BLOCKS   = 12
N_CLASSES  = 10
BATCH_SIZE = 32
TASKS      = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
N_TASKS    = len(TASKS)

_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Weight transfer: ViT MLP → HGF DeepNetwork ────────────────────────────────

def extract_mlp_weights(block):
    l1, l2 = block.mlp[0], block.mlp[3]
    to_np = lambda t: t.detach().cpu().float().numpy()
    return to_np(l1.weight), to_np(l1.bias), to_np(l2.weight), to_np(l2.bias)


def build_hgf_mlp(w1, b1, w2, b2):
    """3-layer HGF (output→hidden→input) with exact forward equivalence to ViT MLP."""
    identity = lambda x: x
    gelu_1 = float(jax.nn.gelu(jnp.array(1.0)))
    net = (
        DeepNetwork()
        .add_layer(EMBED_DIM, add_constant_input=False, coupling_fn=identity)
        .add_layer(MLP_DIM,   add_constant_input=True,  coupling_fn=jax.nn.gelu)
        .add_layer(EMBED_DIM, add_constant_input=True,  coupling_fn=identity)
        .weight_initialisation("he", seed=0)
    )
    ws = list(net.state.weights)
    ws[0] = jnp.concatenate([jnp.array(w2), jnp.array(b2[:, None]) / gelu_1], axis=1)
    ws[1] = jnp.concatenate([jnp.array(w1), jnp.array(b1[:, None])],          axis=1)
    net.state = net.state._replace(weights=tuple(ws))
    return net


def build_all_hgf_blocks(vit):
    return [build_hgf_mlp(*extract_mlp_weights(b)) for b in vit.encoder.layers]


def _hgf_from_torch_mlp(mlp):
    w1 = mlp[0].weight.detach().cpu().float().numpy()
    b1 = mlp[0].bias.detach().cpu().float().numpy()
    w2 = mlp[3].weight.detach().cpu().float().numpy()
    b2 = mlp[3].bias.detach().cpu().float().numpy()
    return build_hgf_mlp(w1, b1, w2, b2)


# ── Linear probe ──────────────────────────────────────────────────────────────

def train_probe_from_cache(n_epochs=10):
    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {CACHE_DIR}. Run vit_hgf_head.py first."
        )
    x_tr = torch.tensor(np.load(CACHE_DIR / "x_train.npy"), dtype=torch.float32)
    y_tr = torch.tensor(np.load(CACHE_DIR / "y_train.npy"), dtype=torch.long)
    probe  = nn.Linear(EMBED_DIM, N_CLASSES)
    opt    = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loader = DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr),
        batch_size=256, shuffle=True,
    )
    probe.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(probe(xb), yb).backward()
            opt.step()
    for p in probe.parameters():
        p.requires_grad_(False)
    probe.eval()
    print(f"  Probe trained ({n_epochs} epochs on 50k cached embeddings).")
    return probe


# ── Block 0-10 forward (used only during one-time cache build) ─────────────────

def _forward_blocks_0_to_K(vit, hgf_blocks, x, stop_before):
    """Run ViT + HGF residuals through blocks 0..(stop_before-1), then stop_before attention only.

    Returns (tokens_post_attn, h_stop) where:
        tokens_post_attn = tokens after self-attention of block stop_before (no MLP residual)
        h_stop           = LN2(tokens_post_attn) = MLP input for block stop_before
    """
    with torch.no_grad():
        tokens = vit._process_input(x)
        B = tokens.shape[0]
        cls = vit.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + vit.encoder.pos_embedding

        for i, block in enumerate(vit.encoder.layers):
            normed = block.ln_1(tokens)
            attn, _ = block.self_attention(normed, normed, normed, need_weights=False)
            tokens = tokens + attn

            if i < stop_before:
                h = block.ln_2(tokens)
                B_, N, D = h.shape
                h_np = h.cpu().float().numpy().reshape(B_ * N, D)
                delta_np = np.asarray(hgf_blocks[i].predict(h_np))
                delta = torch.tensor(delta_np.reshape(B_, N, D),
                                     dtype=tokens.dtype, device=tokens.device)
                tokens = tokens + delta
            else:
                h = block.ln_2(tokens)
                break

    return tokens, h


# ── Cache precomputation ───────────────────────────────────────────────────────

def precompute_cache(vit, hgf_blocks, dataset, batch_size=64, desc=""):
    """Run blocks 0-10 once for all images and cache what block 11 needs.

    Returns:
        tokens_cls : (N, 768)      CLS position of tokens_post_attn at block 11
        h11        : (N, 197, 768) LN2 output = block 11 MLP input (all token positions)
        labels     : (N,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_tokens_cls, all_h11, all_labels = [], [], []
    n = len(loader)
    for i, (imgs, labels) in enumerate(loader):
        print(f"\r  Caching {desc}: {i+1}/{n}", end="", flush=True)
        tokens, h = _forward_blocks_0_to_K(
            vit, hgf_blocks, imgs.to(DEVICE), stop_before=N_BLOCKS - 1
        )
        all_tokens_cls.append(tokens[:, 0].cpu().float().numpy())  # (B, 768) CLS only
        all_h11.append(h.cpu().float().numpy())                     # (B, 197, 768)
        all_labels.append(labels.numpy())
    print(f"\r  Caching {desc}: done ({n} batches)          ")
    return (
        np.concatenate(all_tokens_cls, axis=0),  # (N, 768)
        np.concatenate(all_h11,        axis=0),  # (N, 197, 768)
        np.concatenate(all_labels,     axis=0),  # (N,)
    )


def class_split(tokens_cls, h11, labels, classes, n_max=None):
    """Filter cached arrays to samples of given classes."""
    idx = np.where(np.isin(labels, list(classes)))[0]
    if n_max is not None:
        idx = idx[:n_max]
    return tokens_cls[idx], h11[idx], labels[idx]


# ── Cached block-11 forward ────────────────────────────────────────────────────

def embed_block11(vit, hgf_block11, tokens_cls, h11):
    """Apply block 11 HGF and final LN using cached intermediate activations.

    tokens_cls : (B, 768)      cached CLS of tokens_post_attn
    h11        : (B, 197, 768) cached MLP input for block 11

    vit.encoder.ln is LayerNorm over the embedding dim — applied per token
    independently — so we can apply it to the CLS vector directly.

    Returns (B, 768) numpy array.
    """
    B, N, D = h11.shape
    delta_np  = np.asarray(hgf_block11.predict(h11.reshape(B * N, D)))  # (B*N, 768)
    delta_cls = delta_np.reshape(B, N, D)[:, 0, :]                       # (B, 768)
    cls_raw   = torch.from_numpy(tokens_cls + delta_cls).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        cls_embed = vit.encoder.ln(cls_raw)  # LN over embedding dim only
    return cls_embed.cpu().float().numpy()


def evaluate_cached(vit, hgf_block11, probe, tokens_cls, h11, labels):
    correct = total = 0
    for i in range(0, len(labels), BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)
        cls = embed_block11(vit, hgf_block11, tokens_cls[sl], h11[sl])
        logits = probe(torch.from_numpy(cls))
        correct += (logits.argmax(1).numpy() == labels[sl]).sum()
        total   += len(labels[sl])
    return correct / total


# ── Condition A: frozen ────────────────────────────────────────────────────────

def run_frozen(vit, hgf_block11, probe, test_splits):
    print("\n── Condition A: frozen hybrid ──")
    accs = [evaluate_cached(vit, hgf_block11, probe, *s) for s in test_splits]
    print("  Accuracy: " + " | ".join(f"T{j+1}={a:.3f}" for j, a in enumerate(accs)))
    acc_matrix = np.full((N_TASKS, N_TASKS), np.nan)
    for t in range(N_TASKS):
        for j in range(N_TASKS):
            acc_matrix[t, j] = accs[j]
    return acc_matrix


# ── Condition B: HGF online adaptation ────────────────────────────────────────

def hgf_adapt_step_cached(vit, hgf_block11, probe, tokens_cls, h11, labels, step_size):
    B, N, D = h11.shape
    h_flat   = h11.reshape(B * N, D)
    delta_np = np.asarray(hgf_block11.predict(h_flat))      # (B*N, 768)
    delta_cls_np = delta_np.reshape(B, N, D)[:, 0, :]       # (B, 768)

    # Autograd only through the CLS delta; patch positions contribute zero gradient.
    delta_t  = torch.tensor(delta_cls_np, dtype=torch.float32,
                             device=DEVICE, requires_grad=True)
    cls_raw  = torch.from_numpy(tokens_cls).to(DEVICE, dtype=torch.float32) + delta_t
    cls_embed = vit.encoder.ln(cls_raw).cpu()
    loss = F.cross_entropy(probe(cls_embed),
                           torch.from_numpy(labels.astype(np.int64)))
    loss.backward()

    grad_cls = delta_t.grad.detach().cpu().float().numpy()   # (B, 768)

    # Build (B*N, 768) gradient: CLS at position 0 of each sample, zeros elsewhere.
    grad_full        = np.zeros_like(delta_np)
    grad_full[::N]   = grad_cls                              # token 0 of every sample

    target = delta_np - step_size * grad_full
    hgf_block11.fit(
        h_flat.astype(np.float32),
        target.astype(np.float32),
        lr=step_size,
        learning_kind="precision_weighted",
    )


def _reset_hgf_layer_states(hgf: DeepNetwork) -> None:
    """Reset layer means/precisions to initial values; keep weights."""
    new_layers = tuple(
        LayerState.default(layer.mean.shape[0])
        for layer in hgf.state.layers
    )
    hgf.state = hgf.state._replace(layers=new_layers)


def run_hgf_adapt(vit, hgf_block11, probe, train_splits, test_splits, step_size, rng):
    print(f"\n── Condition B: HGF-adapt block 11 (step_size={step_size}) ──")
    acc_matrix = np.full((N_TASKS, N_TASKS), np.nan)

    for t, (tr_cls, tr_h11, tr_labels) in enumerate(train_splits):
        _reset_hgf_layer_states(hgf_block11)   # fresh precision each task
        w_before = [np.array(w).copy() for w in hgf_block11.state.weights]

        idx = rng.permutation(len(tr_labels))
        for i in range(0, len(idx) - BATCH_SIZE + 1, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]
            hgf_adapt_step_cached(
                vit, hgf_block11, probe,
                tr_cls[bi], tr_h11[bi], tr_labels[bi], step_size,
            )

        delta_w = sum(
            np.linalg.norm(np.array(w) - wb) ** 2
            for w, wb in zip(hgf_block11.state.weights, w_before)
        ) ** 0.5

        for j in range(t + 1):
            acc_matrix[t, j] = evaluate_cached(vit, hgf_block11, probe, *test_splits[j])
        print(
            f"  After task {t+1} [ΔW={delta_w:.4f}]: "
            + " | ".join(f"T{j+1}={acc_matrix[t,j]:.3f}" for j in range(t + 1))
        )
    return acc_matrix


# ── Condition C: Adam fine-tune ────────────────────────────────────────────────

def run_adam_finetune(vit, hgf_blocks_master, probe, train_splits, test_splits,
                      adam_lr, rng):
    print(f"\n── Condition C: Adam fine-tune block 11 MLP (lr={adam_lr}) ──")

    mlp_11 = copy.deepcopy(vit.encoder.layers[-1].mlp).to(DEVICE)
    for p in mlp_11.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(mlp_11.parameters(), lr=adam_lr)

    acc_matrix = np.full((N_TASKS, N_TASKS), np.nan)

    for t, (tr_cls, tr_h11, tr_labels) in enumerate(train_splits):
        w_before = {n: p.detach().cpu().clone() for n, p in mlp_11.named_parameters()}

        mlp_11.train()
        idx = rng.permutation(len(tr_labels))
        for i in range(0, len(idx) - BATCH_SIZE + 1, BATCH_SIZE):
            bi = idx[i:i + BATCH_SIZE]
            # Only CLS position matters for the loss; LN is per-token so we can
            # apply it directly to the CLS vector rather than all 197 tokens.
            h11_t   = torch.from_numpy(tr_h11[bi]).to(DEVICE, dtype=torch.float32)
            cls_pa  = torch.from_numpy(tr_cls[bi]).to(DEVICE, dtype=torch.float32)
            labels_t = torch.from_numpy(tr_labels[bi].astype(np.int64))

            opt.zero_grad()
            delta_cls = mlp_11(h11_t)[:, 0, :]               # (B, 768) CLS position
            cls_raw   = cls_pa + delta_cls
            cls_embed = vit.encoder.ln(cls_raw).cpu()
            F.cross_entropy(probe(cls_embed), labels_t).backward()
            opt.step()

        delta_w = sum(
            (p.detach().cpu() - w_before[n]).norm().item() ** 2
            for n, p in mlp_11.named_parameters()
        ) ** 0.5

        hgf_eval = _hgf_from_torch_mlp(mlp_11)
        for j in range(t + 1):
            acc_matrix[t, j] = evaluate_cached(vit, hgf_eval, probe, *test_splits[j])
        print(
            f"  After task {t+1} [ΔW={delta_w:.4f}]: "
            + " | ".join(f"T{j+1}={acc_matrix[t,j]:.3f}" for j in range(t + 1))
        )

    return acc_matrix


# ── BWT metric ────────────────────────────────────────────────────────────────

def backward_transfer(acc_matrix):
    total, count = 0.0, 0
    for i in range(1, N_TASKS):
        for j in range(i):
            if not np.isnan(acc_matrix[i, j]) and not np.isnan(acc_matrix[j, j]):
                total += acc_matrix[i, j] - acc_matrix[j, j]
                count += 1
    return total / count if count else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--n-train",   type=int,   default=1000,
                        help="Training samples per task (default: 1000)")
    parser.add_argument("--n-eval",    type=int,   default=400,
                        help="Test samples per task (default: 400)")
    parser.add_argument("--adam-lr",   type=float, default=1e-3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    print(f"Device: {DEVICE}")
    print("Loading ViT-B/16 (pretrained ImageNet)...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)

    print(f"Building {N_BLOCKS} HGF MLP blocks from pretrained weights...")
    hgf_blocks_master = build_all_hgf_blocks(vit)

    print("\nTraining linear probe on cached ViT embeddings...")
    probe = train_probe_from_cache(n_epochs=10)

    # ── Build task index lists using .targets (fast, no iteration) ────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=_TRANSFORM)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=_TRANSFORM)

    tr_targets = np.array(train_ds.targets)
    te_targets = np.array(test_ds.targets)

    task_train_idx, task_test_idx = [], []
    for classes in TASKS:
        mask = np.isin(tr_targets, list(classes))
        task_train_idx.extend(np.where(mask)[0][:args.n_train].tolist())
        mask = np.isin(te_targets, list(classes))
        task_test_idx.extend(np.where(mask)[0][:args.n_eval].tolist())

    # ── Precompute block 0-10 activations (one-time cost) ─────────────────────
    print(f"\nPrecomputing block 0-10 activations for "
          f"{len(task_train_idx)} train + {len(task_test_idx)} test samples...")
    tr_cls, tr_h11, tr_lbl = precompute_cache(
        vit, hgf_blocks_master, Subset(train_ds, task_train_idx), desc="train"
    )
    te_cls, te_h11, te_lbl = precompute_cache(
        vit, hgf_blocks_master, Subset(test_ds, task_test_idx),  desc="test"
    )

    train_splits = [class_split(tr_cls, tr_h11, tr_lbl, classes) for classes in TASKS]
    test_splits  = [class_split(te_cls, te_h11, te_lbl, classes) for classes in TASKS]

    print(f"\nTasks: {TASKS} | n_train={args.n_train} | n_eval={args.n_eval}")

    # ── Conditions ────────────────────────────────────────────────────────────
    hgf_A = copy.deepcopy(hgf_blocks_master[-1])
    am_frozen = run_frozen(vit, hgf_A, probe, test_splits)

    hgf_B = copy.deepcopy(hgf_blocks_master[-1])
    am_hgf = run_hgf_adapt(vit, hgf_B, probe, train_splits, test_splits,
                            step_size=args.step_size, rng=rng)

    am_adam = run_adam_finetune(vit, hgf_blocks_master, probe, train_splits, test_splits,
                                adam_lr=args.adam_lr, rng=rng)

    bwt_frozen = backward_transfer(am_frozen)
    bwt_hgf    = backward_transfer(am_hgf)
    bwt_adam   = backward_transfer(am_adam)

    print("\n── Summary ──")
    print(f"  Frozen hybrid BWT : {bwt_frozen:+.4f}")
    print(f"  HGF-adapt BWT     : {bwt_hgf:+.4f}")
    print(f"  Adam fine-tune BWT: {bwt_adam:+.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / "vit_hgf_adapt.csv"
    header = ["condition", "bwt"] + [f"task{j+1}_final_acc" for j in range(N_TASKS)]
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, am, bwt in [
            ("frozen",        am_frozen, bwt_frozen),
            ("hgf_adapt",     am_hgf,    bwt_hgf),
            ("adam_finetune", am_adam,   bwt_adam),
        ]:
            final = [
                round(am[N_TASKS - 1, j], 4) if not np.isnan(am[N_TASKS - 1, j]) else None
                for j in range(N_TASKS)
            ]
            writer.writerow([name, round(bwt, 4)] + final)

    # ── Plot ──────────────────────────────────────────────────────────────────
    conditions = [
        ("Frozen hybrid",           am_frozen, bwt_frozen),
        ("HGF-adapt block 11",      am_hgf,    bwt_hgf),
        ("Adam fine-tune block 11", am_adam,   bwt_adam),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x_ticks = range(1, N_TASKS + 1)
    for ax, (title, am, bwt) in zip(axes, conditions):
        for j in range(N_TASKS):
            vals = [am[i, j] if i >= j and not np.isnan(am[i, j]) else np.nan
                    for i in range(N_TASKS)]
            ax.plot(x_ticks, vals, marker="o", label=f"Task {j+1} {TASKS[j]}")
        ax.set_title(f"{title}\nBWT = {bwt:+.3f}")
        ax.set_xlabel("Tasks trained so far")
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Split-CIFAR-10: frozen ViT-B/16 backbone, online block 11 adaptation")
    fig.tight_layout()
    out = Path(__file__).parent / "vit_hgf_adapt.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot    → {out}")
    print(f"Results → {results_path}")


if __name__ == "__main__":
    main()
