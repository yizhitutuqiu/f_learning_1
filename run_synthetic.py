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
    get_global_train_loader,
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
    total = sum(n for _, n, *_ in results)
    if total == 0:
        return results[0][0] if results else None
    aggregated = []
    for i in range(len(results[0][0])):
        agg_i = np.zeros_like(results[0][0][i])
        for item in results:
            params, n = item[0], item[1]
            agg_i += params[i] * (n / total)
        aggregated.append(agg_i)
    return aggregated


def _squared_norm_list(arr_list):
    """Sum of squared Frobenius norms over a list of arrays."""
    return sum((a.astype(np.float64) ** 2).sum() for a in arr_list)


def dissimilarity(results, global_params):
    """Variance of client updates (gradient dissimilarity): (1/K) * sum_k ||update_k - mean_update||^2."""
    if not results or global_params is None:
        return float("nan")
    K = len(results)
    updates = []
    weights = []
    for (params, n, *_ ) in results:
        upd = [np.asarray(p, dtype=np.float64) - np.asarray(g, dtype=np.float64) for p, g in zip(params, global_params)]
        updates.append(upd)
        weights.append(n)
    total_w = sum(weights)
    if total_w == 0:
        return float("nan")
    mean_upd = []
    for i in range(len(global_params)):
        m = np.zeros_like(global_params[i], dtype=np.float64)
        for j, (upd, w) in enumerate(zip(updates, weights)):
            m += upd[i] * (w / total_w)
        mean_upd.append(m)
    return (1.0 / K) * sum(_squared_norm_list([upd[i] - mean_upd[i] for i in range(len(mean_upd))]) for upd in updates)


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
    parser.add_argument("--drop_percent", type=float, default=0.0,
                        help="Fraction of selected clients as stragglers (0, 0.5, 0.9). Stragglers do 1~num_epochs-1 local epochs.")
    args = parser.parse_args()

    # FedProx dirs use dots: synthetic_0.5_0.5
    dataset_name = args.dataset.replace("_0_5_0_5", "_0.5_0.5")

    torch.manual_seed(1 + args.seed)

    clients, train_data, test_data = load_fedprox_synthetic(dataset_name, args.data_root)
    num_clients = len(clients)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_test_loader = get_global_test_loader(test_data, clients, batch_size=256)
    global_train_loader = get_global_train_loader(train_data, clients, batch_size=256)

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

    def central_train_loss(ndarrays: list):
        """Training loss as in paper: global model loss on full training set (weighted over clients)."""
        model = MCLR().to(device)
        set_parameters(model, ndarrays)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        loss_sum, total = 0.0, 0
        with torch.no_grad():
            for x, y in global_train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += criterion(logits, y).item() * x.size(0)
                total += x.size(0)
        return loss_sum / total if total else 0.0

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
    history[0]["train_loss"] = central_train_loss(current_parameters)
    history[0]["dissimilarity"] = float("nan")
    log(f"  round 0  accuracy: {history[0]['accuracy']:.4f}  loss: {history[0]['loss']:.4f}")
    pbar = tqdm(range(1, args.num_rounds + 1), desc="rounds", unit="rnd", ncols=100)
    for rnd in pbar:
        config = fit_config_fn(rnd)
        selected = random.sample(range(num_clients), min(args.clients_per_round, num_clients))
        # Stragglers: same seed per round for reproducibility (align with FedProx)
        rng = np.random.default_rng(rnd + args.seed * 1000)
        n_active = max(1, round(len(selected) * (1 - args.drop_percent)))
        active_set = set(rng.choice(selected, size=min(n_active, len(selected)), replace=False).tolist())
        results = []
        for idx in selected:
            # FedAvg: only train & aggregate active clients (drop stragglers); FedProx: accept all (variable local_epochs)
            if args.optimizer == "fedavg" and args.drop_percent > 0 and idx not in active_set:
                continue  # FedAvg drops stragglers (paper: "simply drop the slow devices")
            local_epochs = args.num_epochs if idx in active_set else int(rng.integers(1, max(2, args.num_epochs)))
            config_round = {**config, "local_epochs": local_epochs}
            client = _client_fn(idx)
            params, num_samples, metrics = client.fit(current_parameters, config_round)
            results.append((params, num_samples, metrics))
        # Dissimilarity uses global params at round start (before aggregation)
        global_before = current_parameters
        current_parameters = weighted_avg_parameters(results)
        dissim = dissimilarity(results, global_before)
        central_evaluate(rnd, current_parameters)
        # Paper-style training loss: global model on full training set (smooth curve)
        history[-1]["train_loss"] = central_train_loss(current_parameters)
        history[-1]["dissimilarity"] = dissim
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
    drop_suffix = f"_drop{int(round(args.drop_percent * 100))}" if args.drop_percent > 0 else ""
    log_name = f"{args.dataset}_client10_epoch20_{mu_suffix}{drop_suffix}{args.output_suffix}"

    # 1) 配置文件快照（可追溯）
    config_snapshot = vars(args).copy()
    config_snapshot["dataset_resolved"] = dataset_name
    config_snapshot["drop_percent"] = args.drop_percent
    config_path = os.path.join(args.output_dir, log_name + "_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
    log(f"Wrote {config_path}")

    # 2) 训练/评估日志：.jsonl（每行一个 round）
    jsonl_path = os.path.join(args.log_dir, log_name + ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for h in history:
            row = {"round": h["round"], "accuracy": h["accuracy"], "loss": h["loss"]}
            if "train_loss" in h and h["train_loss"] is not None:
                row["train_loss"] = h["train_loss"]
            if "dissimilarity" in h:
                row["dissimilarity"] = h["dissimilarity"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"Wrote {jsonl_path}")

    # 3) 训练/评估日志：.csv
    csv_path = os.path.join(args.log_dir, log_name + ".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("round,accuracy,loss,train_loss,dissimilarity\n")
        for h in history:
            tl = h.get("train_loss")
            diss = h.get("dissimilarity", float("nan"))
            f.write(f"{h['round']},{h['accuracy']:.6f},{h['loss']:.6f},{tl if tl is not None else ''},{diss}\n")
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
