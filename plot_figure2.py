"""
Plot Figure 2 (FedProx paper): data heterogeneity — training loss (top) and dissimilarity (bottom).
No stragglers (drop_percent=0). Four synthetic datasets; two curves: FedAvg (mu=0) vs FedProx (mu>0).
Requires runs with train_loss and dissimilarity in history (run_synthetic.py with default drop_percent=0).
Usage:
  # After running all 4 datasets with mu=0 and mu=1 (e.g. run_all_synthetic.sh)
  python plot_figure2.py --output_dir output
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
MU1_NAME = "FedProx (mu>0)"


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
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    metrics_dir = args.output_dir
    os.makedirs(metrics_dir, exist_ok=True)

    # Top row: training loss
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, ds in enumerate(DATASETS):
        ax = axes[0, idx]
        m0 = load_metrics(metrics_dir, ds, 0)
        m1 = load_metrics(metrics_dir, ds, 1)
        if m0 and m0.get("history"):
            h = m0["history"]
            rnds = [x["round"] for x in h]
            tl = [x.get("train_loss") for x in h]
            if any(t is not None for t in tl):
                rnds_ = [r for r, t in zip(rnds, tl) if t is not None]
                tl_ = [t for t in tl if t is not None]
                ax.plot(rnds_, tl_, "--", lw=2, label=MU0_NAME, color="#17becf")
            else:
                ax.plot(rnds, [x["loss"] for x in h], "--", lw=2, label=MU0_NAME + " (test loss)", color="#17becf")
        if m1 and m1.get("history"):
            h = m1["history"]
            rnds = [x["round"] for x in h]
            tl = [x.get("train_loss") for x in h]
            if any(t is not None for t in tl):
                rnds_ = [r for r, t in zip(rnds, tl) if t is not None]
                tl_ = [t for t in tl if t is not None]
                ax.plot(rnds_, tl_, lw=2, label=MU1_NAME, color="#e377c2")
            else:
                ax.plot(rnds, [x["loss"] for x in h], lw=2, label=MU1_NAME + " (test loss)", color="#e377c2")
        ax.set_title(ds.replace("_", "\n"))
        ax.set_xlabel("Round")
        ax.set_ylabel("Training loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # Bottom row: dissimilarity
    for idx, ds in enumerate(DATASETS):
        ax = axes[1, idx]
        m0 = load_metrics(metrics_dir, ds, 0)
        m1 = load_metrics(metrics_dir, ds, 1)
        for m, style, label, color in [(m0, "--", MU0_NAME, "#17becf"), (m1, "-", MU1_NAME, "#e377c2")]:
            if m and m.get("history"):
                h = m["history"]
                rnds = [x["round"] for x in h]
                diss = [x.get("dissimilarity") for x in h]
                valid = [(r, d) for r, d in zip(rnds, diss) if d is not None and not (isinstance(d, float) and np.isnan(d))]
                if valid:
                    ax.plot([v[0] for v in valid], [v[1] for v in valid], style, lw=2, label=label, color=color)
        ax.set_title(ds.replace("_", "\n"))
        ax.set_xlabel("Round")
        ax.set_ylabel("Dissimilarity\n(var of gradients)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Figure 2: Data heterogeneity — training loss (top) and dissimilarity (bottom)")
    plt.tight_layout()
    out = os.path.join(metrics_dir, "figure2_data_heterogeneity.pdf")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
