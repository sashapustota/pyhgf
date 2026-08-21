"""E0 — dose-response validation of pi*w^2 as a feature-relevance score.

Go/no-go experiment for PRECISION_INTERPRETABILITY.md. Synthetic task with exact
ground-truth relevance per feature, in three groups:

- informative  (f0..f7):   z, the latent that generates the label. relevance = 1.
- pure noise   (f8..f15):  independent of z.                       relevance = 0.
- noisy signal (f16..f23): z + sigma * eps, sigma swept.            relevance = 1/(1+sigma^2).

For each sigma in the sweep, a fresh network is trained and five per-input-feature
attribution scores are computed from the input->hidden weight matrix:

- pi_only          mean(weights_precision_delta, axis=hidden)   -- Finding 1's flat baseline
- pi_w2 (ours)      sum(precision_delta * weights_mean**2, axis=hidden) -- the OBD/Laplace score
- abs_w             sum(|weights_mean|, axis=hidden)
- grad_x_input      mean_samples(|dL/dx * x|), autodiff through the trained network
- post_hoc_fisher   sum(mean_samples(dL/dw)**2, axis=hidden), empirical Fisher diagonal
- true_ablation     accuracy drop when the feature is zeroed at test time

Two metrics per (method, sigma): Spearman rank correlation of the score against the
24-vector of ground-truth relevance, and AUC separating {informative, noisy-signal}
from {pure noise} using the score as a threshold-free classifier. pi_w2 should beat
pi_only on both, and both curves should degrade monotonically as sigma grows.

Only depends on packages already in pyproject.toml (numpy, pandas, matplotlib, jax,
equinox, optax) plus pyhgf itself -- no scipy/sklearn.

Usage
-----
    python experiments/e0_dose_response.py                 # full protocol
    python experiments/e0_dose_response.py --quick          # smoke test (small N, 1 seed)
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from pyhgf.model import DeepNetwork
from pyhgf.utils.vectorized_belief_propagation import (
    batched_prediction_pass,
    prediction_pass,
)

RESULTS_DIR = Path(__file__).parent / "results"

D_GROUP = 8  # features per group; total input width is 3 * D_GROUP
INPUT_LAYER_IDX = 2  # net.state.layers[2] -- see PRECISION_INTERPRETABILITY.md Finding 3

METHODS = (
    "pi_only",
    "pi_w2 (ours)",
    "abs_w",
    "grad_x_input",
    "fisher_only",
    "post_hoc_fisher (fisher*w2)",
    "true_ablation",
)


# --------------------------------------------------------------------------- #
# Dependency-free rank statistics (Spearman rho, Mann-Whitney AUC).
# --------------------------------------------------------------------------- #


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-indexed), ties resolved by their mean rank."""
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


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Network + data.
# --------------------------------------------------------------------------- #


def _lrelu(x):
    return jnp.where(x > 0, x, 0.01 * x)


def _lin(x):
    return x


def build_net(n_input: int, seed: int) -> DeepNetwork:
    """binary(2) <- 16 <- input(n_input), matching PRECISION_INTERPRETABILITY.md."""
    net = DeepNetwork(coupling_fn=_lrelu)
    net.add_layer(size=2, kind="binary", tonic_volatility_vol=-4.0, add_constant_input=False)
    net.add_layer(
        size=16, volatility_parent=True, tonic_volatility_vol=-4.0, add_constant_input=True
    )
    net.add_layer(
        size=n_input,
        volatility_parent=True,
        tonic_volatility_vol=-4.0,
        add_constant_input=False,
        coupling_fn=_lin,
    )
    net.weight_initialisation(strategy="he", key=jax.random.key(seed))
    return net


def make_dataset(rng: np.random.Generator, n: int):
    """One (z, w_true, noise_block, eps) draw, reused across the sigma sweep.

    Returns everything needed to build X(sigma) without re-drawing randomness, so the
    sweep isolates the effect of sigma from sampling noise.
    """
    z = rng.normal(size=(n, D_GROUP))
    w_true = rng.normal(size=D_GROUP)
    y = ((z @ w_true) > 0).astype(int)
    noise_block = rng.normal(size=(n, D_GROUP))
    eps = rng.normal(size=(n, D_GROUP))
    return z, y, noise_block, eps


def build_inputs(z, y, noise_block, eps, sigma: float):
    n = z.shape[0]
    noisy_signal = z + sigma * eps
    x_raw = np.concatenate([z, noise_block, noisy_signal], axis=1)
    x = (x_raw - x_raw.mean(0)) / x_raw.std(0)
    Y = np.zeros((n, 2), dtype=np.float32)
    Y[np.arange(n), y] = 1.0
    relevance = np.concatenate(
        [np.ones(D_GROUP), np.zeros(D_GROUP), np.full(D_GROUP, 1.0 / (1.0 + sigma**2))]
    )
    group_label = np.array([1] * D_GROUP + [0] * D_GROUP + [1] * D_GROUP)
    return x.astype(np.float32), Y, relevance, group_label


# --------------------------------------------------------------------------- #
# Attribution scores.
# --------------------------------------------------------------------------- #


def accuracy(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> float:
    preds = np.asarray(net.predict(X))
    return float((preds.argmax(1) == Y.argmax(1)).mean())


@eqx.filter_jit
def _grad_x_input_raw(state, X, Y):
    def loss(x):
        p = jnp.clip(batched_prediction_pass(state, x), 1e-6, 1 - 1e-6)
        return -jnp.mean(Y * jnp.log(p) + (1 - Y) * jnp.log(1 - p))

    g = jax.grad(loss)(X)
    return jnp.mean(jnp.abs(g * X), axis=0)


def grad_x_input_scores(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.asarray(_grad_x_input_raw(net.state, jnp.asarray(X), jnp.asarray(Y)))


@eqx.filter_jit
def _fisher_diag_raw(state, X, Y, layer_idx: int):
    w0 = state.layers[layer_idx].weights_mean

    def per_sample_loss(w, xi, yi):
        layers = list(state.layers)
        layers[layer_idx] = dataclasses.replace(layers[layer_idx], weights_mean=w)
        net2 = dataclasses.replace(state, layers=tuple(layers))
        p = jnp.clip(prediction_pass(net2, xi), 1e-6, 1 - 1e-6)
        return -jnp.sum(yi * jnp.log(p) + (1 - yi) * jnp.log(1 - p))

    grads = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))(w0, X, Y)
    return jnp.mean(grads**2, axis=0)


def _fisher_diag(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Empirical Fisher diagonal on the input-layer weights, shape (16, D).

    Computed the "everyone else" way: a dedicated backward pass over the whole
    dataset after training, rather than read off an online-tracked belief.
    """
    return np.asarray(
        _fisher_diag_raw(net.state, jnp.asarray(X), jnp.asarray(Y), INPUT_LAYER_IDX)
    )


def fisher_only_scores(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Raw Fisher diagonal, aggregated like pi_only -- expected to be flat/inverted.

    Diagnostic mirror of Finding 1's pi_only result: curvature alone, from either
    source (online belief or post-hoc backward pass), should not rank relevance.
    """
    return _fisher_diag(net, X, Y).mean(axis=0)


def post_hoc_fisher_scores(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fisher-weighted OBD score (fisher * w^2), aggregated like pi_w2.

    This is the actual competitor to pi_w2 for claim (a): the standard
    backprop-computed saliency, requiring a dedicated backward pass pi_w2 gets for
    free from training. Comparing this to fisher_only_scores is what claim (a)
    ("the net's online belief already tracks post-hoc curvature") needs to survive.
    """
    wm = np.asarray(net.state.layers[INPUT_LAYER_IDX].weights_mean)
    return (_fisher_diag(net, X, Y) * wm**2).sum(axis=0)


def true_ablation_scores(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Accuracy drop from zeroing (mean-imputing, X is z-scored) one feature at a time.

    Flagged as underpowered in Finding 3 for single hidden units; kept here as the
    baseline E0 asked for, on input features rather than hidden units.
    """
    base_acc = accuracy(net, X, Y)
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x_ablated = X.copy()
        x_ablated[:, j] = 0.0
        scores[j] = base_acc - accuracy(net, x_ablated, Y)
    return scores


def all_scores(net: DeepNetwork, X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    layer = net.state.layers[INPUT_LAYER_IDX]
    wm = np.asarray(layer.weights_mean)
    wp = np.asarray(layer.weights_precision_delta)
    return {
        "pi_only": wp.mean(axis=0),
        "pi_w2 (ours)": (wp * wm**2).sum(axis=0),
        "abs_w": np.abs(wm).sum(axis=0),
        "grad_x_input": grad_x_input_scores(net, X, Y),
        "fisher_only": fisher_only_scores(net, X, Y),
        "post_hoc_fisher (fisher*w2)": post_hoc_fisher_scores(net, X, Y),
        "true_ablation": true_ablation_scores(net, X, Y),
    }


# --------------------------------------------------------------------------- #
# Sweep.
# --------------------------------------------------------------------------- #


def run_sweep(
    sigmas: list[float], seeds: list[int], n_samples: int, verbose: bool = True
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        z, y, noise_block, eps = make_dataset(rng, n_samples)
        n_input = 3 * D_GROUP

        for sigma in sigmas:
            t0 = time.time()
            X, Y, relevance, group_label = build_inputs(z, y, noise_block, eps, sigma)

            net = build_net(n_input, seed=seed + 1000).fit(
                X,
                Y,
                learning_kind="synaptic_uncertainty",
                learning_kwargs={"window": 200, "prior_variance": 0.02},
                check_gradient_health=False,
            )
            base_acc = accuracy(net, X, Y)
            scores = all_scores(net, X, Y)

            for method in METHODS:
                s = scores[method]
                rows.append(
                    dict(
                        seed=seed,
                        sigma=sigma,
                        method=method,
                        spearman=spearman(s, relevance),
                        auc=auc_score(group_label, s),
                        base_acc=base_acc,
                    )
                )

            if verbose:
                dt = time.time() - t0
                print(
                    f"seed={seed} sigma={sigma:5.2f} acc={base_acc:.3f} "
                    f"({dt:.1f}s)"
                )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "sigma"])[["spearman", "auc"]]
        .agg(["mean", "std"])
        .round(3)
    )


def plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    agg = df.groupby(["method", "sigma"])[["spearman", "auc"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        sub = agg[agg.method == method].sort_values("sigma")
        axes[0].plot(sub.sigma, sub.spearman, marker="o", label=method)
        axes[1].plot(sub.sigma, sub.auc, marker="o", label=method)
    axes[0].set_xlabel("noise sigma (noisy-signal group)")
    axes[0].set_ylabel("Spearman rho vs ground-truth relevance")
    axes[0].set_title("Ranking quality")
    axes[1].set_xlabel("noise sigma (noisy-signal group)")
    axes[1].set_ylabel("AUC: {informative, noisy} vs {pure noise}")
    axes[1].set_title("Separability")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="small N, one seed, few sigmas -- smoke test only"
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.quick:
        n_samples = args.n_samples or 400
        seeds = args.seeds or [0]
        sigmas = args.sigmas or [0.0, 1.0, 3.0, 8.0]
    else:
        n_samples = args.n_samples or 4000
        seeds = args.seeds or [0, 1, 2]
        sigmas = args.sigmas or [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(sigmas, seeds, n_samples)
    df.to_csv(args.output_dir / "e0_dose_response.csv", index=False)
    print(f"\nwrote {args.output_dir / 'e0_dose_response.csv'}")

    print("\n" + summarize(df).to_string())

    plot(df, args.output_dir / "e0_dose_response.png")

    # Go/no-go: pi_w2 should beat pi_only on both metrics, at every sigma.
    pivot_rho = df.groupby(["method", "sigma"])["spearman"].mean().unstack("method")
    pivot_auc = df.groupby(["method", "sigma"])["auc"].mean().unstack("method")
    rho_ok = (pivot_rho["pi_w2 (ours)"] > pivot_rho["pi_only"]).all()
    auc_ok = (pivot_auc["pi_w2 (ours)"] > pivot_auc["pi_only"]).all()
    print(f"\npi_w2 beats pi_only on Spearman at every sigma: {rho_ok}")
    print(f"pi_w2 beats pi_only on AUC at every sigma:       {auc_ok}")


if __name__ == "__main__":
    main()
