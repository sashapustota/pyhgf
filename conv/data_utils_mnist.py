"""MNIST data utility — downloads via torchvision, returns numpy arrays.

Data is cached to <repo_root>/data/mnist/ and is gitignored.

Usage:
    from data_utils_mnist import load_mnist
    X_tr, y_tr, X_te, y_te = load_mnist(DATA_DIR)

Returns:
    X_tr: (60000, 784) float32, normalised to [0, 1]
    y_tr: (60000,)     int64
    X_te: (10000, 784) float32
    y_te: (10000,)     int64
"""
import os
import numpy as np


def load_mnist(data_dir: str):
    import torchvision
    import torchvision.transforms as T

    os.makedirs(data_dir, exist_ok=True)

    transform = T.Compose([T.ToTensor()])

    train_ds = torchvision.datasets.MNIST(
        data_dir, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform)

    def to_numpy(ds):
        X = np.stack([img.numpy().ravel() for img, _ in ds]).astype(np.float32)
        y = np.array([label for _, label in ds], dtype=np.int64)
        return X, y

    print("Loading MNIST train...", flush=True)
    X_tr, y_tr = to_numpy(train_ds)
    print("Loading MNIST test...",  flush=True)
    X_te, y_te = to_numpy(test_ds)
    print(f"  Train: {X_tr.shape} | Test: {X_te.shape}\n", flush=True)

    return X_tr, y_tr, X_te, y_te
