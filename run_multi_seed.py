"""
Run one experiment with multiple seeds, then plot mean ± std (optional, for 提交内容 图表).
Usage:
  python run_multi_seed.py --dataset synthetic_1_1 --optimizer fedprox --mu 1 --seeds 0,1,2
Output: logs/*.jsonl, output/*_metrics.json, output/*_config.json, and output/accuracy_vs_rounds_mean_std.pdf, loss_vs_rounds_mean_std.pdf
"""
import argparse
import json
import os
import subprocess
import sys
import glob
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_1_1")
    parser.add_argument("--optimizer", type=str, default="fedprox", choices=["fedavg", "fedprox"])
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for seed in seeds:
        cmd = [
            sys.executable, "run_synthetic.py",
            "--dataset", args.dataset,
            "--optimizer", args.optimizer,
            "--mu", str(args.mu),
            "--seed", str(seed),
            "--num_rounds", str(args.num_rounds),
            "--output_dir", args.output_dir,
            "--log_dir", args.log_dir,
            "--output_suffix", f"_seed{seed}",
        ]
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

    # Aggregate metrics by seed and plot mean ± std
    mu_suffix = "mu0" if args.mu == 0 else "mu1"
    log_name_base = f"{args.dataset}_client10_epoch20_{mu_suffix}"
    histories = []
    for seed in seeds:
        path = os.path.join(args.output_dir, log_name_base + f"_seed{seed}_metrics.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("history"):
                histories.append(d["history"])
        except Exception:
            pass

    if len(histories) < 2:
        print("Need at least 2 seeds to plot mean±std; skipping.")
        return

    # Build round-wise mean and std
    max_rnd = max(len(h) for h in histories)
    acc_mean = np.nan * np.ones(max_rnd)
    acc_std = np.nan * np.ones(max_rnd)
    loss_mean = np.nan * np.ones(max_rnd)
    loss_std = np.nan * np.ones(max_rnd)
    rounds = list(range(max_rnd))
    for r in range(max_rnd):
        accs = [h[r]["accuracy"] for h in histories if r < len(h)]
        loss_vals = [h[r]["loss"] for h in histories if r < len(h)]
        if accs:
            acc_mean[r] = np.mean(accs)
            acc_std[r] = np.std(accs)
        if loss_vals:
            loss_mean[r] = np.mean(loss_vals)
            loss_std[r] = np.std(loss_vals)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(rounds, acc_mean, color="#e377c2", lw=2, label="mean accuracy")
    ax.fill_between(rounds, acc_mean - acc_std, acc_mean + acc_std, color="#e377c2", alpha=0.3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy vs rounds ({args.dataset}, {args.optimizer}, seeds={args.seeds})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "accuracy_vs_rounds_mean_std.pdf"))
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(rounds, loss_mean, color="#17becf", lw=2, label="mean loss")
    ax.fill_between(rounds, loss_mean - loss_std, loss_mean + loss_std, color="#17becf", alpha=0.3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test loss")
    ax.set_title(f"Loss vs rounds ({args.dataset}, {args.optimizer}, seeds={args.seeds})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_vs_rounds_mean_std.pdf"))
    plt.close()
    print(f"Saved accuracy_vs_rounds_mean_std.pdf and loss_vs_rounds_mean_std.pdf in {args.output_dir}")


if __name__ == "__main__":
    main()
