"""
Run FedAvg / FedProx on FedProx synthetic datasets using Flower + PyTorch.
Uses a custom simulation loop (no flwr.simulation.run_simulation) so it works with any flwr version.
Usage:
  python run_synthetic.py --dataset synthetic_iid --optimizer fedavg
  python run_synthetic.py --dataset synthetic_1_1 --optimizer fedprox --mu 1.0
Output:
  - logs/   : per-round text logs (*.txt)
  - output/ : final results (*_metrics.json)
"""
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from client import client_fn
from dataset import (
    FEDPROX_DATA_ROOT,
    get_global_test_loader,
    load_fedprox_synthetic,
)
from model import MCLR, get_parameters, set_parameters

# Defaults matching FedProx run_fedavg.sh / run_fedprox.sh
NUM_ROUNDS = 200
CLIENTS_PER_ROUND = 10
BATCH_SIZE = 10
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
EVAL_EVERY = 1


def weighted_avg_parameters(results):
    """Aggregate client (params, num_samples) into one parameter list (FedAvg/FedProx same)."""
    total = sum(n for _, n in results)
    if total == 0:
        return results[0][0] if results else None
    aggregated = []
    for i in range(len(results[0][0])):
        agg_i = np.zeros_like(results[0][0][i])
        for (params, n) in results:
            agg_i += params[i] * (n / total)
        aggregated.append(agg_i)
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_iid",
                        help="One of: synthetic_iid, synthetic_0_0, synthetic_0.5_0.5, synthetic_1_1")
    parser.add_argument("--optimizer", type=str, default="fedprox", choices=["fedavg", "fedprox"])
    parser.add_argument("--mu", type=float, default=1.0, help="Proximal mu (FedProx); 0 = FedAvg")
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--clients_per_round", type=int, default=CLIENTS_PER_ROUND)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for per-round text logs (.txt)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for final results and metrics JSON")
    parser.add_argument("--data_root", type=str, default=FEDPROX_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_suffix", type=str, default="", help="e.g. _seed1 for multi-seed runs")
    args = parser.parse_args()

    # FedProx dirs use dots: synthetic_0.5_0.5
    dataset_name = args.dataset.replace("_0_5_0_5", "_0.5_0.5")

    torch.manual_seed(1 + args.seed)

    clients, train_data, test_data = load_fedprox_synthetic(dataset_name, args.data_root)
    num_clients = len(clients)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_test_loader = get_global_test_loader(test_data, clients, batch_size=256)

    # Client factory: Flower passes client index (int); map to FedProx client id (e.g. f_00000)
    def _client_fn(cid):
        if isinstance(cid, int):
            cid = clients[cid]
        return client_fn(
            cid=cid,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            data_root=args.data_root,
        )

    # Initial parameters (list of ndarray)
    init_model = MCLR().to(device)
    current_parameters = get_parameters(init_model)

    # Central evaluation and history for logging
    history = []

    def central_evaluate(server_round: int, ndarrays: list):
        model = MCLR().to(device)
        set_parameters(model, ndarrays)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in global_test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += criterion(logits, y).item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        loss_avg = loss_sum / total if total else 0.0
        accuracy = correct / total if total else 0.0
        history.append({"round": server_round, "accuracy": accuracy, "loss": loss_avg})
        return loss_avg, {"accuracy": accuracy, "loss": loss_avg}

    def fit_config_fn(rnd: int):
        cfg = {"learning_rate": args.learning_rate}
        if args.optimizer == "fedprox":
            cfg["proximal_mu"] = args.mu
            cfg["proximal-mu"] = args.mu
        return cfg

    # Custom simulation loop (no flwr.simulation dependency)
    def log(msg):
        print(msg, flush=True)

    log(f"Running {args.optimizer} on {dataset_name}, mu={args.mu}, rounds={args.num_rounds}")
    random.seed(1 + args.seed)
    np.random.seed(2 + args.seed)

    central_evaluate(0, current_parameters)
    log(f"  round 0  accuracy: {history[0]['accuracy']:.4f}  loss: {history[0]['loss']:.4f}")
    pbar = tqdm(range(1, args.num_rounds + 1), desc="rounds", unit="rnd", ncols=100)
    for rnd in pbar:
        config = fit_config_fn(rnd)
        selected = random.sample(range(num_clients), min(args.clients_per_round, num_clients))
        results = []
        for idx in selected:
            cid = clients[idx]
            client = _client_fn(cid)
            params, num_samples, _ = client.fit(current_parameters, config)
            results.append((params, num_samples))
        current_parameters = weighted_avg_parameters(results)
        central_evaluate(rnd, current_parameters)
        acc, loss_val = history[-1]["accuracy"], history[-1]["loss"]
        pbar.set_postfix(acc=f"{acc:.4f}", loss=f"{loss_val:.4f}")

    # Final metrics (last round from history if available)
    final_metrics = history[-1] if history else {}
    final_accuracy = final_metrics.get("accuracy", 0.0)
    final_loss = final_metrics.get("loss", 0.0)
    log(f"Final test accuracy: {final_accuracy:.4f}, loss: {final_loss:.4f}")

    import json
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    mu_suffix = "mu0" if args.mu == 0 else "mu1"
    log_name = f"{args.dataset}_client10_epoch20_{mu_suffix}{args.output_suffix}"

    # 1) 配置文件快照（可追溯）
    config_snapshot = vars(args).copy()
    config_snapshot["dataset_resolved"] = dataset_name
    config_path = os.path.join(args.output_dir, log_name + "_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
    log(f"Wrote {config_path}")

    # 2) 训练/评估日志：.jsonl（每行一个 round）
    jsonl_path = os.path.join(args.log_dir, log_name + ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps({"round": h["round"], "accuracy": h["accuracy"], "loss": h["loss"]}, ensure_ascii=False) + "\n")
    log(f"Wrote {jsonl_path}")

    # 3) 训练/评估日志：.csv
    csv_path = os.path.join(args.log_dir, log_name + ".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("round,accuracy,loss\n")
        for h in history:
            f.write(f"{h['round']},{h['accuracy']:.6f},{h['loss']:.6f}\n")
    log(f"Wrote {csv_path}")

    # 4) 兼容 FedProx plot 的 .txt
    log_path = os.path.join(args.log_dir, log_name + ".txt")
    with open(log_path, "w") as f:
        for h in history:
            rnd, acc, loss = h["round"], h.get("accuracy", 0), h.get("loss", 0)
            f.write(f"At round {rnd} accuracy: {acc}\n")
            f.write(f"At round {rnd} training loss: {loss}\n")
    log(f"Wrote {log_path}")

    # 5) 最终结果与 history -> output/
    metrics_path = os.path.join(args.output_dir, log_name + "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "optimizer": args.optimizer,
            "mu": args.mu,
            "num_rounds": args.num_rounds,
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "history": history,
        }, f, indent=2, ensure_ascii=False)
    log(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
