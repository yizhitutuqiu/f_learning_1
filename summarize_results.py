"""
Collect final_accuracy / final_loss from output/*_metrics.json into output/summary.csv and summary.json.
"""
import argparse
import json
import os
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    out_dir = args.output_dir
    pattern = os.path.join(out_dir, "*_client10_epoch20_mu*_metrics.json")
    files = sorted(glob.glob(pattern))
    rows = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        rows.append({
            "dataset": d.get("dataset", ""),
            "optimizer": d.get("optimizer", ""),
            "mu": d.get("mu", 0),
            "num_rounds": d.get("num_rounds", 0),
            "final_accuracy": d.get("final_accuracy", 0),
            "final_loss": d.get("final_loss", 0),
        })
    if not rows:
        print(f"No *_metrics.json found in {out_dir}")
        return
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {summary_path} ({len(rows)} experiments)")
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,optimizer,mu,num_rounds,final_accuracy,final_loss\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['optimizer']},{r['mu']},{r['num_rounds']},{r['final_accuracy']:.6f},{r['final_loss']:.6f}\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
