"""E1 -- Fisher agreement between pyhgf's online belief and a separate backprop MLP.

Tests claim (a) in PRECISION_INTERPRETABILITY.md: "the net knows its own importance"
-- correlate `weights_precision_delta`, tracked online during training by a pyhgf
DeepNetwork, against a post-hoc empirical Fisher diagonal computed from a completely
independent, standard-backprop MLP (different weight init, different training
procedure, no pyhgf machinery at all).

This is a stronger test than E0's internal Fisher check: hidden units in the two
networks have no shared identity (arbitrary permutation from independent training), so
the only well-posed comparison is at the per-input-pixel level, aggregating each
network's raw (unweighted) curvature over its own hidden axis -- exactly `pi_only`'s
aggregation, on both sides. Per Finding 1, raw curvature is driven substantially by
input statistics (~E[x^2]), which both networks see identically -- so unlike E0's
ranking experiments, agreement here is *expected* if pi_only is really tracking a
Fisher-like quantity, independent of whether that quantity ranks label-relevance.

Data: FashionMNIST, downloaded once from the canonical zalandoresearch mirror and
cached under experiments/data/ (gitignored).

Usage
-----
    python experiments/e1_fisher_agreement.py                 # full protocol
    python experiments/e1_fisher_agreement.py --quick          # smoke test
"""

from __future__ import annotations

import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pyhgf.model import DeepNetwork

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data" / "fashion_mnist"
BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = {
    "images": "train-images-idx3-ubyte.gz",
    "labels": "train-labels-idx1-ubyte.gz",
}

N_PIXELS = 28 * 28
N_CLASSES = 10
INPUT_LAYER_IDX = 2  # pyhgf net: layers = [categorical(10), hidden, input(784)]


# --------------------------------------------------------------------------- #
# Data.
# --------------------------------------------------------------------------- #


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / name
    if not dest.exists():
        print(f"downloading {BASE_URL + name} ...")
        urllib.request.urlretrieve(BASE_URL + name, dest)
    return dest


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} for images file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} for labels file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_fashion_mnist(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    images = _read_idx_images(_download(FILES["images"])).astype(np.float32) / 255.0
    labels = _read_idx_labels(_download(FILES["labels"])).astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(n_samples, len(images)), replace=False)
    X = images[idx]
    # Center per pixel but scale by one *global* std, not per-pixel std: FashionMNIST
    # has near-constant-zero border pixels, and per-pixel z-scoring divides by a
    # near-zero std there, blowing a few images' rounding noise into huge outliers
    # that both networks then pick up on identically -- an inflated, spurious
    # agreement, not real signal.
    X = (X - X.mean(0)) / (X.std() + 1e-6)
    Y = np.zeros((len(idx), N_CLASSES), dtype=np.float32)
    Y[np.arange(len(idx)), labels[idx]] = 1.0
    return X.astype(np.float32), Y


# --------------------------------------------------------------------------- #
# Dependency-free Spearman (see experiments/e0_dose_response.py for the AUC twin).
# --------------------------------------------------------------------------- #


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks + 1.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(np.asarray(a, dtype=float)), _rankdata(np.asarray(b, dtype=float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# pyhgf net -- categorical output, synaptic_uncertainty for weights_precision_delta.
# --------------------------------------------------------------------------- #


def build_pyhgf_net(hidden: int, seed: int) -> DeepNetwork:
    net = DeepNetwork(coupling_fn=jax.nn.leaky_relu)
    net.add_layer(size=N_CLASSES, kind="categorical")
    net.add_layer(size=hidden, tonic_volatility_vol=-4.0)
    net.add_layer(
        size=N_PIXELS,
        add_constant_input=False,
        coupling_fn=lambda x: x,
        tonic_volatility_vol=-4.0,
    )
    net.weight_initialisation(strategy="he", key=jax.random.key(seed))
    return net


def pyhgf_curvature_scores(net: DeepNetwork) -> np.ndarray:
    """Raw (unweighted) precision, aggregated like pi_only -- (N_PIXELS,)."""
    wp = np.asarray(net.state.layers[INPUT_LAYER_IDX].weights_precision_delta)
    return wp.mean(axis=0)


# --------------------------------------------------------------------------- #
# Independent backprop MLP -- plain jax/optax, no pyhgf machinery.
# --------------------------------------------------------------------------- #


def init_mlp(hidden: int, key: jax.Array) -> dict:
    k1, k2 = jax.random.split(key)
    scale1 = np.sqrt(2.0 / N_PIXELS)
    scale2 = np.sqrt(2.0 / hidden)
    return {
        "w1": jax.random.normal(k1, (N_PIXELS, hidden)) * scale1,
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k2, (hidden, N_CLASSES)) * scale2,
        "b2": jnp.zeros(N_CLASSES),
    }


def mlp_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    h = jax.nn.leaky_relu(x @ params["w1"] + params["b1"])
    logits = h @ params["w2"] + params["b2"]
    return jax.nn.log_softmax(logits)


def mlp_loss(params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    log_p = mlp_forward(params, x)
    return -jnp.mean(jnp.sum(y * log_p, axis=-1))


def train_mlp(X: np.ndarray, Y: np.ndarray, hidden: int, epochs: int, seed: int) -> dict:
    params = init_mlp(hidden, jax.random.key(seed))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    grad_fn = jax.jit(jax.grad(mlp_loss))

    n = X.shape[0]
    batch_size = 128
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            grads = grad_fn(params, jnp.asarray(X[batch]), jnp.asarray(Y[batch]))
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        acc = float(
            (mlp_forward(params, jnp.asarray(X)).argmax(1) == Y.argmax(1)).mean()
        )
        print(f"  mlp epoch {epoch}: train acc {acc:.3f}")
    return params


def mlp_fisher_scores(params: dict, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Empirical Fisher diagonal on w1, aggregated like pi_only -- (N_PIXELS,)."""

    def per_sample_loss(w1, xi, yi):
        h = jax.nn.leaky_relu(xi @ w1 + params["b1"])
        logits = h @ params["w2"] + params["b2"]
        log_p = jax.nn.log_softmax(logits)
        return -jnp.sum(yi * log_p)

    grad_fn = jax.jit(jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0)))
    # Chunk over samples to bound peak memory: (chunk, N_PIXELS, hidden) per chunk.
    chunks = []
    chunk_size = 500
    for start in range(0, X.shape[0], chunk_size):
        xb = jnp.asarray(X[start : start + chunk_size])
        yb = jnp.asarray(Y[start : start + chunk_size])
        g = grad_fn(params["w1"], xb, yb)  # (chunk, N_PIXELS, hidden)
        chunks.append(np.asarray(g**2).sum(axis=0))
    fisher_sum = np.sum(chunks, axis=0)  # (N_PIXELS, hidden)
    fisher_diag = fisher_sum / X.shape[0]
    return fisher_diag.mean(axis=1)  # aggregate over hidden axis, like pi_only


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #


def plot(pi_scores, fisher_scores, rho, rho_high_var, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.scatter(fisher_scores, pi_scores, s=8, alpha=0.5)
    ax.set_xlabel("backprop MLP: empirical Fisher diagonal (pixel-aggregated)")
    ax.set_ylabel("pyhgf: weights_precision_delta (pixel-aggregated)")
    ax.set_title(f"Fisher agreement (Spearman rho={rho:.2f})")

    ax = axes[1]
    im = ax.imshow(pi_scores.reshape(28, 28), cmap="viridis")
    ax.set_title("pyhgf curvature map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    im = ax.imshow(fisher_scores.reshape(28, 28), cmap="viridis")
    ax.set_title("backprop MLP Fisher map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"rho (high-variance pixels only) = {rho_high_var:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small N, few epochs -- smoke test only")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.quick:
        n_samples = args.n_samples or 1000
        hidden = args.hidden or 32
        epochs = args.epochs or 3
    else:
        n_samples = args.n_samples or 8000
        hidden = args.hidden or 64
        epochs = args.epochs or 15

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {n_samples} FashionMNIST samples ...")
    X, Y = load_fashion_mnist(n_samples, seed=args.seed)

    print("training pyhgf net (synaptic_uncertainty) ...")
    net = build_pyhgf_net(hidden, seed=args.seed).fit(
        X,
        Y,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 200, "prior_variance": 0.02},
        check_gradient_health=False,
    )
    acc = float(
        (np.asarray(net.predict(X)).argmax(1) == Y.argmax(1)).mean()
    )
    print(f"  pyhgf net train acc: {acc:.3f}")
    pi_scores = pyhgf_curvature_scores(net)

    print("training independent backprop MLP ...")
    mlp_params = train_mlp(X, Y, hidden, epochs, seed=args.seed + 1)
    print("computing empirical Fisher diagonal (dedicated backward pass) ...")
    fisher_scores = mlp_fisher_scores(mlp_params, X, Y)

    rho = spearman(pi_scores, fisher_scores)

    # Robustness check: restrict to the higher-variance (non-background) half of
    # pixels, since FashionMNIST's all-zero border trivially agrees on both sides.
    pixel_var = X.var(axis=0)
    high_var = pixel_var > np.median(pixel_var)
    rho_high_var = spearman(pi_scores[high_var], fisher_scores[high_var])

    plot(pi_scores, fisher_scores, rho, rho_high_var, args.output_dir / "e1_fisher_agreement.png")

    print(f"\nSpearman(pi_only, post_hoc_fisher) over all {N_PIXELS} pixels: {rho:.3f}")
    print(f"Spearman restricted to high-variance (non-border) pixels:      {rho_high_var:.3f}")
    print(f"\nClaim (a) gate (rho > 0.3 on high-variance pixels): {'PASS' if rho_high_var > 0.3 else 'FAIL'}")


if __name__ == "__main__":
    main()
