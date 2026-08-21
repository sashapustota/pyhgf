"""E3 -- the "money experiment": per-neuron change-point localisation.

Tests PRECISION_INTERPRETABILITY.md's novel claim (b): on a drift/task-switch stream,
mean_vol spikes and precision drops, localized to the units carrying the changed
features. Requires volatility parents (Finding 2) -- this is the thing no post-hoc
method gives you without rerunning the whole pipeline at every point in time.

Design: a streaming reversal-learning task (classic paradigm in the HGF /
computational-psychiatry literature). Same 24-feature structure as E0:

- group A (f0..f7):   informative, = z
- group B (f8..f15):  pure noise, always uninformative
- group C (f16..f23): noisy signal, = z + sigma*eps (sigma fixed, not swept here)

y = sign(z @ w_true) for the first half of the stream. At the changepoint t=T1,
w_true -> -w_true (total label reversal): every decision boundary the network has
learned for groups A and C is now backwards, while group B was never informative and
stays that way. This gives a clean two-way ground truth split of hidden units:

- "affected" units: top-K by |weight| reliance on groups A+C, snapshotted from the
  state right at T1 (before the reversal's effects can contaminate the snapshot).
- "control" units: bottom-K by that same score (units that leaned on noise).

Metric: for each unit, the change in mean_vol / precision from a pre-T1 baseline
window to a post-T1 response window. The localisation claim is that this response is
larger for "affected" units than "control" units -- not just that *some* global signal
spikes at T1 (classical single-node HGF change-point detection already does that; the
novel part is that many per-neuron signals move differentially and line up with which
weights actually became wrong).

Usage
-----
    python experiments/e3_reversal_changepoint.py                 # full protocol
    python experiments/e3_reversal_changepoint.py --quick          # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from pyhgf.model import DeepNetwork

RESULTS_DIR = Path(__file__).parent / "results"

D_GROUP = 8
D_INPUT = 3 * D_GROUP
HIDDEN_SIZE = 16
INPUT_LAYER_IDX = 2
HIDDEN_LAYER_IDX = 1
SIGMA_C = 1.0  # fixed noise level for the noisy-signal group (not swept here)
TOP_K = 5  # units per side (affected / control) out of HIDDEN_SIZE


def _lrelu(x):
    return jnp.where(x > 0, x, 0.01 * x)


def _lin(x):
    return x


def build_net(seed: int) -> DeepNetwork:
    net = DeepNetwork(coupling_fn=_lrelu)
    net.add_layer(size=2, kind="binary", tonic_volatility_vol=-4.0, add_constant_input=False)
    net.add_layer(
        size=HIDDEN_SIZE,
        volatility_parent=True,
        tonic_volatility_vol=-4.0,
        add_constant_input=True,
    )
    net.add_layer(
        size=D_INPUT,
        volatility_parent=True,
        tonic_volatility_vol=-4.0,
        add_constant_input=False,
        coupling_fn=_lin,
    )
    net.weight_initialisation(strategy="he", key=jax.random.key(seed))
    return net


def make_stream(rng: np.random.Generator, n_total: int, t1: int):
    """Build the reversal stream: same z/eps throughout, w_true flips sign at t1."""
    z = rng.normal(size=(n_total, D_GROUP))
    noise_block = rng.normal(size=(n_total, D_GROUP))
    eps = rng.normal(size=(n_total, D_GROUP))
    noisy_signal = z + SIGMA_C * eps
    x_raw = np.concatenate([z, noise_block, noisy_signal], axis=1)
    X = ((x_raw - x_raw.mean(0)) / x_raw.std(0)).astype(np.float32)

    w_true = rng.normal(size=D_GROUP)
    w_true /= np.linalg.norm(w_true)
    y_pre = (z[:t1] @ w_true) > 0
    y_post = (z[t1:] @ -w_true) > 0
    y = np.concatenate([y_pre, y_post]).astype(int)

    Y = np.zeros((n_total, 2), dtype=np.float32)
    Y[np.arange(n_total), y] = 1.0
    return X, Y


def reliance_scores(weights_mean: np.ndarray) -> dict[str, np.ndarray]:
    """Per-hidden-unit sum(|w|) reliance on each input feature group. Shape (16,)."""
    return {
        "A": np.abs(weights_mean[:, 0:D_GROUP]).sum(axis=1),
        "B": np.abs(weights_mean[:, D_GROUP : 2 * D_GROUP]).sum(axis=1),
        "C": np.abs(weights_mean[:, 2 * D_GROUP : 3 * D_GROUP]).sum(axis=1),
    }


def run_one_seed(seed: int, n_total: int, t1: int, window: int) -> dict:
    rng = np.random.default_rng(seed)
    X, Y = make_stream(rng, n_total, t1)

    net = build_net(seed=seed + 2000)
    optimizer = optax.sgd(0.02)
    record = ("precision", "mean_vol", "precision_vol")

    # Phase 1: pre-reversal. Snapshot ground-truth unit reliance right here, before
    # the reversal's error signal can touch the weights.
    net.fit(
        X[:t1],
        Y[:t1],
        optimizer=optimizer,
        learning_kind="precision_weighted",
        record=record,
        check_gradient_health=False,
    )
    traj_pre = net.trajectories
    preds_pre = np.asarray(net.predictions)
    reliance = reliance_scores(np.asarray(net.state.layers[INPUT_LAYER_IDX].weights_mean))

    # Phase 2: post-reversal, continuing from the same state/opt_state.
    net.fit(
        X[t1:],
        Y[t1:],
        optimizer=optimizer,
        learning_kind="precision_weighted",
        record=record,
        check_gradient_health=False,
    )
    traj_post = net.trajectories
    preds_post = np.asarray(net.predictions)

    precision = np.concatenate(
        [np.asarray(traj_pre["precision"][HIDDEN_LAYER_IDX]),
         np.asarray(traj_post["precision"][HIDDEN_LAYER_IDX])],
        axis=0,
    )  # (n_total, 16)
    mean_vol = np.concatenate(
        [np.asarray(traj_pre["mean_vol"][HIDDEN_LAYER_IDX]),
         np.asarray(traj_post["mean_vol"][HIDDEN_LAYER_IDX])],
        axis=0,
    )
    preds = np.concatenate([preds_pre, preds_post], axis=0)
    correct = preds.argmax(axis=1) == Y.argmax(axis=1)

    baseline = slice(max(0, t1 - window), t1)
    response = slice(t1, min(n_total, t1 + window))

    unit_mean_vol_delta = mean_vol[response].mean(axis=0) - mean_vol[baseline].mean(axis=0)
    unit_precision_drop = precision[baseline].mean(axis=0) - precision[response].mean(axis=0)

    combined = reliance["A"] + reliance["C"]
    order = np.argsort(combined)
    control_units = order[:TOP_K]
    affected_units = order[-TOP_K:]

    return dict(
        seed=seed,
        acc_baseline=float(correct[baseline].mean()),
        acc_response=float(correct[response].mean()),
        pop_mean_vol_delta=float(unit_mean_vol_delta.mean()),
        pop_precision_drop=float(unit_precision_drop.mean()),
        affected_mean_vol_delta=float(unit_mean_vol_delta[affected_units].mean()),
        control_mean_vol_delta=float(unit_mean_vol_delta[control_units].mean()),
        affected_precision_drop=float(unit_precision_drop[affected_units].mean()),
        control_precision_drop=float(unit_precision_drop[control_units].mean()),
        # kept for plotting the localisation picture of the first seed
        _mean_vol=mean_vol if seed == 0 else None,
        _precision=precision if seed == 0 else None,
        _reliance_combined=combined if seed == 0 else None,
        _affected_units=affected_units if seed == 0 else None,
        _control_units=control_units if seed == 0 else None,
    )


def plot(rows: list[dict], n_total: int, t1: int, window: int, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    row0 = rows[0]
    mean_vol, precision = row0["_mean_vol"], row0["_precision"]
    affected, control = row0["_affected_units"], row0["_control_units"]

    lo, hi = max(0, t1 - 4 * window), min(n_total, t1 + 4 * window)
    t = np.arange(lo, hi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for u in affected:
        ax.plot(t, mean_vol[lo:hi, u], color="tab:red", alpha=0.5, linewidth=1)
    for u in control:
        ax.plot(t, mean_vol[lo:hi, u], color="tab:blue", alpha=0.5, linewidth=1)
    ax.axvline(t1, color="grey", linestyle="--", linewidth=1)
    ax.plot([], [], color="tab:red", label="affected units (A+C reliant)")
    ax.plot([], [], color="tab:blue", label="control units (noise reliant)")
    ax.set_xlabel("stream step")
    ax.set_ylabel("mean_vol")
    ax.set_title("Seed 0: per-unit mean_vol around the reversal")
    ax.legend(fontsize=8)

    ax = axes[1]
    for u in affected:
        ax.plot(t, precision[lo:hi, u], color="tab:red", alpha=0.5, linewidth=1)
    for u in control:
        ax.plot(t, precision[lo:hi, u], color="tab:blue", alpha=0.5, linewidth=1)
    ax.axvline(t1, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("stream step")
    ax.set_ylabel("precision")
    ax.set_title("Seed 0: per-unit precision around the reversal")

    ax = axes[2]
    df = pd.DataFrame(rows)
    seeds = df["seed"].to_numpy()
    width = 0.35
    xpos = np.arange(len(seeds))
    ax.bar(xpos - width / 2, df["affected_mean_vol_delta"], width, label="affected", color="tab:red")
    ax.bar(xpos + width / 2, df["control_mean_vol_delta"], width, label="control", color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("mean_vol delta (response - baseline)")
    ax.set_title("Localisation effect, per seed")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small stream, one seed -- smoke test only")
    parser.add_argument("--n-total", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.quick:
        n_total = args.n_total or 800
        window = args.window or 60
        seeds = args.seeds or [0]
    else:
        n_total = args.n_total or 6000
        window = args.window or 150
        seeds = args.seeds or [0, 1, 2, 3, 4]
    t1 = n_total // 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        row = run_one_seed(seed, n_total, t1, window)
        rows.append(row)
        print(
            f"seed={seed} acc {row['acc_baseline']:.2f}->{row['acc_response']:.2f}  "
            f"pop mean_vol delta={row['pop_mean_vol_delta']:+.4f}  "
            f"affected={row['affected_mean_vol_delta']:+.4f}  "
            f"control={row['control_mean_vol_delta']:+.4f}"
        )

    plot(rows, n_total, t1, window, args.output_dir / "e3_reversal_changepoint.png")

    df = pd.DataFrame(rows).drop(
        columns=["_mean_vol", "_precision", "_reliance_combined", "_affected_units", "_control_units"]
    )
    df.to_csv(args.output_dir / "e3_reversal_changepoint.csv", index=False)
    print(f"\nwrote {args.output_dir / 'e3_reversal_changepoint.csv'}")
    print("\n" + df.to_string(index=False))

    acc_drop_ok = (df["acc_response"] < df["acc_baseline"] - 0.2).all()
    global_spike_ok = (df["pop_mean_vol_delta"] > 0).all()
    localisation_ok = (df["affected_mean_vol_delta"] > df["control_mean_vol_delta"]).all()
    n_local = (df["affected_mean_vol_delta"] > df["control_mean_vol_delta"]).sum()

    print(f"\nReversal is a real surprise (accuracy drops >0.2 at every seed): {acc_drop_ok}")
    print(f"Global mean_vol spikes at every seed:                            {global_spike_ok}")
    print(
        f"Localisation (affected > control mean_vol delta): {n_local}/{len(df)} seeds "
        f"({'PASS' if localisation_ok else 'partial' if n_local > len(df) // 2 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
