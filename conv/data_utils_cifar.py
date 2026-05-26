"""CIFAR-10 data utility — downloads via torchvision, returns numpy arrays.

Data is cached to <repo_root>/data/cifar10/ and is gitignored.

Usage:
    from data_utils_cifar import load_cifar10
    X_tr, y_tr, X_te, y_te = load_cifar10(DATA_DIR)

Returns:
    X_tr: (50000, 3, 32, 32) float32, per-channel normalised
    y_tr: (50000,)            int64
    X_te: (10000, 3, 32, 32) float32
    y_te: (10000,)            int64

For FC use, flatten X after loading: X_tr.reshape(len(X_tr), -1)
"""
import os
import numpy as np

def load_cifar10(data_dir: str):
    import torchvision
    import torchvision.transforms as T

    os.makedirs(data_dir, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2023, 0.1994, 0.2010),
        ),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform)

    def to_numpy(ds):
        X = np.stack([img.numpy() for img, _ in ds]).astype(np.float32)
        y = np.array([label for _, label in ds], dtype=np.int64)
        return X, y

    print("Loading CIFAR-10 train...", flush=True)
    X_tr, y_tr = to_numpy(train_ds)
    print("Loading CIFAR-10 test...",  flush=True)
    X_te, y_te = to_numpy(test_ds)
    print(f"  Train: {X_tr.shape} | Test: {X_te.shape}\n", flush=True)

    return X_tr, y_tr, X_te, y_te


def augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random horizontal flip + random crop with padding 4 (CIFAR standard augmentation)."""
    N, C, H, W = X.shape
    X = X.copy()

    # Random horizontal flip
    flip = rng.random(N) < 0.5
    X[flip] = X[flip, :, :, ::-1]

    # Random crop: pad by 4, then crop back to 32×32
    pad = 4
    X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    tops  = rng.integers(0, 2 * pad, size=N)
    lefts = rng.integers(0, 2 * pad, size=N)
    X = np.stack([X_pad[i, :, tops[i]:tops[i] + H, lefts[i]:lefts[i] + W] for i in range(N)])

    return X
