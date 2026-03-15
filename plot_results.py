"""
Plot loss and accuracy from output_dir/*_metrics.json (output of run_synthetic.py).
Saves loss.pdf, accuracy.pdf, accuracy_loss_grid.pdf to output_dir.
"""
import argparse
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["synthetic_iid", "synthetic_0_0", "synthetic_0.5_0.5", "synthetic_1_1"]
MU0_NAME = "FedAvg (mu=0)"
MU1_NAME = "FedProx (mu=1)"


def load_metrics(metrics_dir: str, dataset: str, mu: float):
    mu_suffix = "mu0" if mu == 0 else "mu1"
    pattern = os.path.join(metrics_dir, f"{dataset}_client10_epoch20_{mu_suffix}_metrics.json")
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory with *_metrics.json and where to save PDFs")
    parser.add_argument("--log_dir", type=str, default="logs", help="Unused for plot; for compatibility")
    args = parser.parse_args()
    metrics_dir = args.output_dir
    os.makedirs(metrics_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        m0 = load_metrics(metrics_dir, ds, 0)
        m1 = load_metrics(metrics_dir, ds, 1)
        if m0 and m0.get("history"):
            rnds = [h["round"] for h in m0["history"]]
            loss0 = [h["loss"] for h in m0["history"]]
            acc0 = [h["accuracy"] for h in m0["history"]]
            ax.plot(rnds, loss0, "--", lw=2, label=MU0_NAME, color="#17becf")
            ax.plot(rnds, acc0, "--", lw=2, alpha=0.7, color="#17becf")
        if m1 and m1.get("history"):
            rnds = [h["round"] for h in m1["history"]]
            loss1 = [h["loss"] for h in m1["history"]]
            acc1 = [h["accuracy"] for h in m1["history"]]
            ax.plot(rnds, loss1, lw=2, label=MU1_NAME, color="#e377c2")
            ax.plot(rnds, acc1, lw=2, alpha=0.7, color="#e377c2")
        ax.set_title(ds.replace("_", "\n"))
        ax.set_xlabel("Round")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("FedProx vs FedAvg (Flower + PyTorch)")
    plt.tight_layout()
    grid_path = os.path.join(metrics_dir, "accuracy_loss_grid.pdf")
    plt.savefig(grid_path)
    plt.close()
    print(f"Saved {grid_path}")

    for metric, ylabel in [("loss", "Training / Test loss"), ("accuracy", "Test accuracy")]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for idx, ds in enumerate(DATASETS):
            ax = axes[idx]
            m0 = load_metrics(metrics_dir, ds, 0)
            m1 = load_metrics(metrics_dir, ds, 1)
            if m0 and m0.get("history"):
                rnds = [h["round"] for h in m0["history"]]
                vals = [h[metric] for h in m0["history"]]
                ax.plot(rnds, vals, "--", lw=2, label=MU0_NAME, color="#17becf")
            if m1 and m1.get("history"):
                rnds = [h["round"] for h in m1["history"]]
                vals = [h[metric] for h in m1["history"]]
                ax.plot(rnds, vals, lw=2, label=MU1_NAME, color="#e377c2")
            ax.set_title(ds.replace("_", "\n"))
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"{'Test accuracy vs rounds' if metric == 'accuracy' else 'Test loss vs rounds'}")
        plt.tight_layout()
        pdf_path = os.path.join(metrics_dir, f"{metric}_vs_rounds.pdf")
        plt.savefig(pdf_path)
        plt.savefig(os.path.join(metrics_dir, f"{metric}.pdf"))
        plt.close()
        print(f"Saved {pdf_path} (and {metric}.pdf)")


if __name__ == "__main__":
    main()
