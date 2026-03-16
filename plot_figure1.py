"""
Plot Figure 1 (FedProx paper): systems heterogeneity — 0% / 50% / 90% stragglers.
Panels: 0%, 50%, 90% stragglers; each panel: FedAvg (mu=0) vs FedProx (mu>0) training loss.
Requires runs with --drop_percent 0, 0.5, 0.9 and mu=0, mu=1 on the same dataset (e.g. synthetic_1_1).
Usage:
  python run_synthetic.py --dataset synthetic_1_1 --optimizer fedavg --mu 0 --drop_percent 0
  python run_synthetic.py --dataset synthetic_1_1 --optimizer fedprox --mu 1 --drop_percent 0
  ... (repeat for drop_percent 0.5, 0.9)
  python plot_figure1.py --dataset synthetic_1_1 --output_dir output
"""
import argparse
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MU0_NAME = "FedAvg / FedProx (mu=0)"
MU1_NAME = "FedProx (mu>0)"


def load_metrics(metrics_dir: str, dataset: str, mu: float, drop_percent: float):
    mu_suffix = "mu0" if mu == 0 else "mu1"
    drop = int(round(drop_percent * 100))
    if drop == 0:
        pattern = os.path.join(metrics_dir, f"{dataset}_client10_epoch20_{mu_suffix}_metrics.json")
    else:
        pattern = os.path.join(metrics_dir, f"{dataset}_client10_epoch20_{mu_suffix}_drop{drop}_metrics.json")
    # also match without _drop0 in filename (legacy runs)
    if drop == 0:
        files = glob.glob(pattern)
        if not files:
            alt = os.path.join(metrics_dir, f"{dataset}_client10_epoch20_{mu_suffix}_drop0_metrics.json")
            if os.path.isfile(alt):
                files = [alt]
    else:
        files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset", type=str, default="synthetic_1_1",
                        help="Dataset used for Figure 1 (e.g. synthetic_1_1)")
    args = parser.parse_args()
    metrics_dir = args.output_dir
    os.makedirs(metrics_dir, exist_ok=True)

    drops = [0.0, 0.5, 0.9]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for col, drop in enumerate(drops):
        ax = axes[col]
        m0 = load_metrics(metrics_dir, args.dataset, 0.0, drop)
        m1 = load_metrics(metrics_dir, args.dataset, 1.0, drop)
        if m0 and m0.get("history"):
            h = m0["history"]
            rnds = [x["round"] for x in h]
            train_loss = [x.get("train_loss") for x in h]
            if any(t is not None for t in train_loss):
                rnds_ = [r for r, t in zip(rnds, train_loss) if t is not None]
                tl_ = [t for t in train_loss if t is not None]
                ax.plot(rnds_, tl_, "--", lw=2, label=MU0_NAME, color="#17becf")
            else:
                loss = [x["loss"] for x in h]
                ax.plot(rnds, loss, "--", lw=2, label=MU0_NAME + " (test loss)", color="#17becf")
        if m1 and m1.get("history"):
            h = m1["history"]
            rnds = [x["round"] for x in h]
            train_loss = [x.get("train_loss") for x in h]
            if any(t is not None for t in train_loss):
                rnds_ = [r for r, t in zip(rnds, train_loss) if t is not None]
                tl_ = [t for t in train_loss if t is not None]
                ax.plot(rnds_, tl_, lw=2, label=MU1_NAME, color="#e377c2")
            else:
                loss = [x["loss"] for x in h]
                ax.plot(rnds, loss, lw=2, label=MU1_NAME + " (test loss)", color="#e377c2")
        ax.set_title(f"{int(drop*100)}% stragglers")
        ax.set_xlabel("Round")
        ax.set_ylabel("Training loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Figure 1: FedProx vs FedAvg under systems heterogeneity (stragglers)")
    plt.tight_layout()
    out = os.path.join(metrics_dir, "figure1_stragglers.pdf")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
