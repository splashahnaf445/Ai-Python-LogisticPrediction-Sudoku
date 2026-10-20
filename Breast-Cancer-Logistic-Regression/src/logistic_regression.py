"""Binary logistic regression from scratch (NumPy only) on the WDBC dataset.

Pipeline: load -> 70/15/15 split -> z-score (train statistics only) ->
per-sample SGD for several learning rates -> pick the best lr on the
validation set -> report test metrics once -> plot training loss.

Usage:
    python src/logistic_regression.py            # run and show the plot
    python src/logistic_regression.py --no-show  # run headless
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]


def load_wdbc_dataset(filepath):
    """Read wdbc.data. Column 0 (ID) is dropped, column 1 (M/B) becomes 1/0,
    the remaining 30 columns are the numeric features."""
    X_list, y_list = [], []
    with open(filepath, mode="r") as file:
        for row in csv.reader(file):
            if not row:
                continue
            y_list.append(1 if row[1].strip() == "M" else 0)
            X_list.append([float(val) for val in row[2:]])
    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int32)


def train_val_test_split(X, y, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Shuffle once with a fixed seed, then cut into train / validation / test."""
    np.random.seed(seed)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


def standardize_features(X_train, X_val, X_test):
    """z-score every feature using the training mean/std only (no data leakage)."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def sigmoid(z):
    """Logistic function, clipped to avoid overflow in exp."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logistic_regression(X_train, y_train, lr, max_iter=500):
    """Train with per-sample stochastic gradient descent.

    A bias column of ones is appended, theta starts as small random values,
    and for every sample: h = sigmoid(x . theta), theta -= lr * x * (h - y).
    Returns the weights (bias last) and the mean training loss per epoch.
    """
    num_samples, num_features = X_train.shape
    X_biased = np.hstack((X_train, np.ones((num_samples, 1))))

    np.random.seed(42)
    theta = np.random.randn(num_features + 1) * 0.01

    train_loss_history = []
    for _ in range(max_iter):
        total_epoch_loss = 0.0
        for i in range(num_samples):
            x_i, y_i = X_biased[i], y_train[i]
            h = sigmoid(np.dot(x_i, theta))

            eps = 1e-15
            h_clipped = np.clip(h, eps, 1.0 - eps)
            total_epoch_loss += -y_i * np.log(h_clipped) - (1.0 - y_i) * np.log(1.0 - h_clipped)

            theta = theta - lr * (x_i * (h - y_i))
        train_loss_history.append(total_epoch_loss / num_samples)
    return theta, train_loss_history


def predict(X, theta):
    """Class 1 (malignant) when the predicted probability is at least 0.5."""
    X_biased = np.hstack((X, np.ones((X.shape[0], 1))))
    return (sigmoid(np.dot(X_biased, theta)) >= 0.5).astype(int)


def compute_accuracy(y_true, y_pred):
    return (np.sum(y_true == y_pred) / len(y_true)) * 100.0


def confusion_metrics(y_true, y_pred):
    """Confusion counts plus precision / recall / F1 for the malignant class."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "wdbc.data"), help="path to wdbc.data")
    parser.add_argument("--outdir", default=str(ROOT / "results"), help="where to write the plot and metrics")
    parser.add_argument("--max-iter", type=int, default=500, help="epochs per learning rate")
    parser.add_argument("--no-show", action="store_true", help="save the plot without opening a window")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("[Task 1] Loading raw WDBC dataset and executing 70/15/15 split...")
    X, y = load_wdbc_dataset(args.data)
    log(f"Total samples: {X.shape[0]} | Total features: {X.shape[1]}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    log(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}")
    X_train, X_val, X_test = standardize_features(X_train, X_val, X_test)

    results, trained_models, histories = [], {}, {}
    log("\n[Task 2 & 3] Training models across learning rates...")
    for lr in LEARNING_RATES:
        theta, loss_history = train_logistic_regression(X_train, y_train, lr=lr, max_iter=args.max_iter)
        val_acc = compute_accuracy(y_val, predict(X_val, theta))
        results.append((lr, val_acc))
        trained_models[lr] = theta
        histories[lr] = loss_history

    log("\n" + "=" * 46)
    log(f"{'lr':<12} | {'val_acc':<10} | {'final train loss':<16}")
    log("-" * 46)
    for lr, val_acc in results:
        log(f"{lr:<12} | {val_acc:>7.2f}%   | {histories[lr][-1]:<16.4f}")
    log("=" * 46)

    # max() keeps the first of equal scores, so ties go to the larger learning rate
    best_lr, best_val_acc = max(results, key=lambda item: item[1])
    log(f"\nBest Learning Rate: {best_lr} (Validation Accuracy: {best_val_acc:.2f}%)")

    test_preds = predict(X_test, trained_models[best_lr])
    test_acc = compute_accuracy(y_test, test_preds)
    m = confusion_metrics(y_test, test_preds)
    log(f"Final Test Accuracy (lr={best_lr}, max_iter={args.max_iter}): {test_acc:.2f}%")
    log(f"Confusion matrix (malignant = positive): TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    log(f"Malignant precision={m['precision']:.4f} recall={m['recall']:.4f} F1={m['f1']:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.max_iter + 1), histories[best_lr], color="#1f77b4", linewidth=2, label=f"SGD (lr = {best_lr})")
    plt.title(f"Train Loss vs Epoch (Optimal lr = {best_lr})", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch (Iteration)", fontsize=11)
    plt.ylabel("Training Loss (Binary Cross-Entropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = outdir / "train_loss_vs_epoch.png"
    plt.savefig(plot_path, dpi=300)
    log(f"\nSaved plot to {outdir.name}/{plot_path.name}")
    (outdir / "metrics.txt").write_text("\n".join(lines) + "\n")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
