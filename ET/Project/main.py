#!/usr/bin/env python3
"""
rmsprop_2d_contour.py

Experiment:
 - Synthetic regression with 2 input features (x1, x2)
 - Linear model: y = w1*x1 + w2*x2 + b
 - Compare: Vanilla RMSProp  vs  RMSProp + Bias-Correction (2nd moment) + Momentum (1st moment)
 - Visuals:
     * Contour plot of loss over (w1, w2) with optimizer trajectories overlaid
     * 3D surface of the same loss with trajectories drawn in 3D
     * Four line charts with metrics (train loss, val loss, grad norm, update norm)
 - Saves results (PNGs + CSV) under ./results/<timestamp>/

Usage:
    python rmsprop_2d_contour.py
"""

import os
import time
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import LogNorm


# -------------------------
# Utilities
# -------------------------
SEED = 42
np.random.seed(SEED)


def now_str():
    return time.strftime("%Y%m%d-%H%M%S")


def makedirs(path):
    os.makedirs(path, exist_ok=True)


# -------------------------
# Data: 2-feature synthetic regression
# -------------------------
def make_2d_regression(n_samples=1000, noise=5.0, random_state=SEED):
    """
    Generate synthetic data with 2 input features.
    Ground truth: y = 3.0 * x1 - 2.0 * x2 + 1.0 + noise
    Returns: X (n,2), y (n,1), true_params dict
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2) * 2.0  # spread inputs to make landscape interesting
    # true coefficients
    w_true = np.array([3.0, -2.0])
    b_true = 1.0
    y = X.dot(w_true.reshape(-1, 1)) + b_true + rng.randn(n_samples, 1) * noise
    return X, y, {"w_true": w_true.copy(), "b_true": float(b_true)}


# -------------------------
# Linear model (2 features)
# -------------------------
class LinearModel2D:
    def __init__(self, init_scale=0.1):
        rng = np.random.RandomState(SEED)
        # parameter vector w shape (2,), bias scalar
        self.w = rng.randn(2) * init_scale
        self.b = float(0.0)

    def predict(self, X):
        # X: (n,2); return (n,1)
        return X.dot(self.w.reshape(-1, 1)) + self.b

    def loss_and_grads(self, X, y):
        """
        Compute MSE loss and gradients for full-batch data.
        Returns: loss (float), grads dict {"w": array(2,), "b": scalar}
        """
        n = X.shape[0]
        y_pred = self.predict(X)  # (n,1)
        err = y_pred - y  # (n,1)
        loss = float(np.mean(err ** 2))
        # grads:
        # dL/dw = (2/n) * X^T (y_pred - y)  -> shape (2,1) -> squeeze
        grad_w = (2.0 / n) * (X.T.dot(err)).reshape(-1)  # (2,)
        grad_b = float((2.0 / n) * np.sum(err))
        return loss, {"w": grad_w, "b": grad_b}

    def get_params(self):
        return {"w": self.w.copy(), "b": float(self.b)}

    def set_params(self, params: Dict[str, np.ndarray]):
        self.w = params["w"].copy()
        self.b = float(params["b"])


# -------------------------
# Optimizers
# -------------------------
class RMSPropVanilla:
    def __init__(self, lr=1e-2, beta=0.9, eps=1e-8):
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.s = {"w": np.zeros(2), "b": 0.0}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.t += 1
        updates = {}
        # w
        self.s["w"] = self.beta * self.s["w"] + (1 - self.beta) * (grads["w"] ** 2)
        delta_w = - self.lr * grads["w"] / (np.sqrt(self.s["w"]) + self.eps)
        params["w"] = params["w"] + delta_w
        updates["w"] = delta_w.copy()
        # b
        self.s["b"] = self.beta * self.s["b"] + (1 - self.beta) * (grads["b"] ** 2)
        delta_b = - self.lr * grads["b"] / (np.sqrt(self.s["b"]) + self.eps)
        params["b"] = params["b"] + delta_b
        updates["b"] = float(delta_b)
        return params, updates


class RMSPropBCMomentum:
    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, bias_correction_first=True):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.bias_correction_first = bool(bias_correction_first)
        self.m = {"w": np.zeros(2), "b": 0.0}
        self.s = {"w": np.zeros(2), "b": 0.0}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.t += 1
        updates = {}
        denom1 = (1.0 - self.beta1 ** self.t) if self.bias_correction_first else 1.0
        denom2 = (1.0 - self.beta2 ** self.t)
        # w
        self.m["w"] = self.beta1 * self.m["w"] + (1 - self.beta1) * grads["w"]
        self.s["w"] = self.beta2 * self.s["w"] + (1 - self.beta2) * (grads["w"] ** 2)
        m_hat_w = self.m["w"] / denom1
        s_hat_w = self.s["w"] / denom2
        delta_w = - self.lr * m_hat_w / (np.sqrt(s_hat_w) + self.eps)
        params["w"] = params["w"] + delta_w
        updates["w"] = delta_w.copy()
        # b
        self.m["b"] = self.beta1 * self.m["b"] + (1 - self.beta1) * grads["b"]
        self.s["b"] = self.beta2 * self.s["b"] + (1 - self.beta2) * (grads["b"] ** 2)
        m_hat_b = self.m["b"] / denom1
        s_hat_b = self.s["b"] / denom2
        delta_b = - self.lr * m_hat_b / (np.sqrt(s_hat_b) + self.eps)
        params["b"] = params["b"] + float(delta_b)
        updates["b"] = float(delta_b)
        return params, updates


# -------------------------
# Loss surface over (w1, w2)
# -------------------------
def compute_loss_grid(X, y, b_fixed, w1_range, w2_range):
    """
    Compute MSE loss for each (w1,w2) pair, fixing bias to b_fixed.
    Returns meshgrid W1, W2 and loss grid (same shape).
    """
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1, dtype=float)
    n = X.shape[0]
    # Vectorized evaluation
    # For each grid point, compute predictions for all X and MSE
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            y_pred = X.dot(w.reshape(-1, 1)) + b_fixed  # (n,1)
            Z[i, j] = float(np.mean((y_pred - y) ** 2))
    return W1, W2, Z


# -------------------------
# Experiment runner (full-batch)
# -------------------------
def run_training(X_train, y_train, X_val, y_val, optimizer_cls, init_params, hyperparams, epochs=100):
    """
    Train linear model with given optimizer class.
    Returns:
      - metrics dict with lists
      - trajectory list of params per epoch (w1,w2)
      - final model params
    """
    model = LinearModel2D()
    model.w = init_params["w"].copy()
    model.b = float(init_params["b"])
    params = model.get_params()

    optimizer = optimizer_cls(**hyperparams)

    metrics = {"epoch": [], "train_loss": [], "val_loss": [], "val_r2": [], "grad_norm": [], "update_norm": []}
    traj = []  # store (w1,w2) per epoch

    for epoch in range(1, epochs + 1):
        loss, grads = model.loss_and_grads(X_train, y_train)
        grad_norm = np.sqrt(np.sum(grads["w"] ** 2) + grads["b"] ** 2)

        # step
        old_params = {"w": params["w"].copy(), "b": float(params["b"])}
        params, updates = optimizer.step(params, grads)
        # apply to model
        model.set_params(params)

        update_norm = np.sqrt(np.sum(updates["w"] ** 2) + updates["b"] ** 2)

        # validation
        y_val_pred = model.predict(X_val)
        val_loss = float(np.mean((y_val_pred - y_val) ** 2))
        val_r2 = float(r2_score(y_val, y_val_pred))

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_r2"].append(val_r2)
        metrics["grad_norm"].append(grad_norm)
        metrics["update_norm"].append(update_norm)
        traj.append(params["w"].copy())

        # simple console log
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"[{optimizer_cls.__name__}] Epoch {epoch}/{epochs} | train_loss={loss:.6f} val_loss={val_loss:.6f} val_r2={val_r2:.4f} grad_norm={grad_norm:.6f} update_norm={update_norm:.6f}")

    return metrics, np.array(traj), params


# -------------------------
# Plotting: contour + trajectories + 3D surface
# -------------------------
def plot_contour_and_3d(W1, W2, Z, traj_a, traj_b, true_w, out_dir):
    makedirs(out_dir)
    # Contour plot
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    # contour levels use log spacing to show large dynamic range
    levels = np.logspace(np.log10(Z.max() * 1e-6 + 1e-12), np.log10(Z.max() + 1e-12), 40)
    cs = ax1.contourf(W1, W2, Z, levels=levels, norm=LogNorm(), cmap="viridis")

    cbar = fig.colorbar(cs, ax=ax1, pad=0.01)
    cbar.set_label("MSE Loss (log scale)")

    # plot optimizer trajectories (w1,w2)
    traj_a = np.array(traj_a)
    traj_b = np.array(traj_b)
    ax1.plot(traj_a[:, 0], traj_a[:, 1], 'o-', color='orange', markersize=4, label='Vanilla RMSProp')
    ax1.plot(traj_b[:, 0], traj_b[:, 1], 's-', color='cyan', markersize=4, label='RMSProp+BC+Momentum')
    # mark start and end
    ax1.scatter(traj_a[0, 0], traj_a[0, 1], marker='D', color='orange', s=80, label='Start (Vanilla)')
    ax1.scatter(traj_b[0, 0], traj_b[0, 1], marker='D', color='cyan', s=80, label='Start (Improved)')
    ax1.scatter(true_w[0], true_w[1], marker='*', color='red', s=120, label='True w (ground truth)')
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.set_title("Loss Contour over (w1, w2) with optimizer trajectories")
    ax1.legend(loc='upper right')

    # 3D surface with trajectories
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.9, rstride=1, cstride=1, linewidth=0, antialiased=False)
    # lift trajectories by looking up Z value at each (w1,w2) point (approx)
    def z_at(w):
        # find nearest grid index (assumes uniform grid)
        # linear interpolation would be nicer but nearest is fine for plotting trajectory
        w1_vals = W1[0, :]
        w2_vals = W2[:, 0]
        i = np.searchsorted(w2_vals, w[1])
        j = np.searchsorted(w1_vals, w[0])
        i = np.clip(i, 0, W2.shape[0]-1)
        j = np.clip(j, 0, W1.shape[1]-1)
        return Z[i, j]

    za = [z_at(w) for w in traj_a]
    zb = [z_at(w) for w in traj_b]
    ax2.plot(traj_a[:, 0], traj_a[:, 1], za, color='orange', marker='o', label='Vanilla RMSProp')
    ax2.plot(traj_b[:, 0], traj_b[:, 1], zb, color='cyan', marker='s', label='RMSProp+BC+Momentum')
    ax2.scatter(true_w[0], true_w[1], float(Z.mean()), color='red', s=80, marker='*', label='True w')
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.set_zlabel("MSE Loss")
    ax2.set_title("3D Loss Surface and optimizer paths")
    ax2.view_init(elev=35, azim=-60)
    ax2.legend()

    plt.tight_layout()
    out_png = os.path.join(out_dir, "loss_contour_3d_trajectories.png")
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"Saved contour+3D plot to: {out_png}")


# -------------------------
# Plot metrics lines
# -------------------------
def plot_metrics(metrics_a, metrics_b, labels, out_dir):
    makedirs(out_dir)
    epochs = metrics_a["epoch"]
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_a["train_loss"], 'o-', label=f'{labels[0]} Train Loss')
    plt.plot(epochs, metrics_b["train_loss"], 's-', label=f'{labels[1]} Train Loss')
    plt.title("Train Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_a["val_loss"], 'o-', label=f'{labels[0]} Val Loss')
    plt.plot(epochs, metrics_b["val_loss"], 's-', label=f'{labels[1]} Val Loss')
    plt.title("Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_a["val_r2"], 'o-', label=f'{labels[0]} Val R^2')
    plt.plot(epochs, metrics_b["val_r2"], 's-', label=f'{labels[1]} Val R^2')
    plt.title("Val R^2")
    plt.xlabel("Epoch")
    plt.ylabel("R^2")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics_a["grad_norm"], 'o-', label=f'{labels[0]} Grad Norm')
    plt.plot(epochs, metrics_b["grad_norm"], 's-', label=f'{labels[1]} Grad Norm')
    plt.plot(epochs, metrics_a["update_norm"], 'o--', alpha=0.7, label=f'{labels[0]} Update Norm')
    plt.plot(epochs, metrics_b["update_norm"], 's--', alpha=0.7, label=f'{labels[1]} Update Norm')
    plt.title("Grad Norm & Update Norm")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "metrics_comparison.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved metrics plot to: {out_png}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--beta_rms", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--noise", type=float, default=6.0)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_id = f"rmsprop_2d_{now_str()}"
    out_dir = os.path.join(args.results_dir, run_id)
    makedirs(out_dir)

    # 1) data
    X, y, true_params = make_2d_regression(n_samples=args.n_samples, noise=args.noise, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 2) fixed grid for loss surface over (w1, w2)
    # choose ranges centered around true w to show basin + some outskirts
    w1_true, w2_true = true_params["w_true"]
    span = 6.0
    w1_range = np.linspace(w1_true - span, w1_true + span, 120)
    w2_range = np.linspace(w2_true - span, w2_true + span, 120)
    # compute loss grid keeping bias fixed at true bias for a clear static landscape
    W1, W2, Z = compute_loss_grid(X_train, y_train, b_fixed=true_params["b_true"], w1_range=w1_range, w2_range=w2_range)

    # 3) initial params (same start for both optimizers)
    init_params = {"w": np.array([-5.0, 5.0]), "b": 0.0}  # deliberately far from true to show trajectory

    # 4) run baseline (vanilla RMSProp) - full-batch for clean trajectories
    baseline_hyper = {"lr": args.lr, "beta": args.beta_rms, "eps": args.eps}
    print("\n=== Running Vanilla RMSProp ===")
    metrics_vanilla, traj_vanilla, final_vanilla = run_training(X_train, y_train, X_val, y_val,
                                                                RMSPropVanilla, init_params, baseline_hyper,
                                                                epochs=args.epochs)

    # 5) run improved (RMSProp + BC + Momentum) - same init
    improved_hyper = {"lr": args.lr, "beta1": args.beta1, "beta2": args.beta2, "eps": args.eps, "bias_correction_first": True}
    print("\n=== Running RMSProp + Bias-Correction + Momentum ===")
    metrics_improved, traj_improved, final_improved = run_training(X_train, y_train, X_val, y_val,
                                                                   RMSPropBCMomentum, init_params, improved_hyper,
                                                                   epochs=args.epochs)

    # 6) Save metrics CSVs
    makedirs(out_dir)
    pd.DataFrame(metrics_vanilla).to_csv(os.path.join(out_dir, "metrics_vanilla.csv"), index=False)
    pd.DataFrame(metrics_improved).to_csv(os.path.join(out_dir, "metrics_improved.csv"), index=False)

    # 7) Contour + 3D plot (overlay both trajectories)
    plot_contour_and_3d(W1, W2, Z, traj_vanilla, traj_improved, true_params["w_true"], out_dir)

    # 8) metrics plots
    plot_metrics(metrics_vanilla, metrics_improved, ("Vanilla RMSProp", "RMSProp+BC+Momentum"), out_dir)

    # 9) Save summary
    summary = {
        "metric": ["final_train_loss", "final_val_loss", "final_val_r2", "final_grad_norm", "final_update_norm"],
        "vanilla": [metrics_vanilla["train_loss"][-1], metrics_vanilla["val_loss"][-1], metrics_vanilla["val_r2"][-1],
                    metrics_vanilla["grad_norm"][-1], metrics_vanilla["update_norm"][-1]],
        "improved": [metrics_improved["train_loss"][-1], metrics_improved["val_loss"][-1], metrics_improved["val_r2"][-1],
                     metrics_improved["grad_norm"][-1], metrics_improved["update_norm"][-1]]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\nSummary:\n", df_summary.to_string(index=False))
    print(f"\nAll outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
