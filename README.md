# FedProx 复现（Flower + PyTorch）

单机仿真复现论文 [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)（MLSys 2020）的 synthetic 实验，FedAvg vs FedProx。

---

## 1. 环境安装

```bash
cd f_learning
pip install -r requirements.txt
```

依赖：`flwr`、`torch`、`numpy`、`matplotlib`、`tqdm`（见 `requirements.txt`）。

**数据与任务**：Synthetic 数据集为 **60 维特征 → 10 类分类**（线性模型 MCLR），数据由 FedProx 官方脚本合成，用于控制数据/系统异质性。从同目录上级的 `FedProx/data/` 加载（需含 `synthetic_iid`、`synthetic_0_0`、`synthetic_0.5_0.5`、`synthetic_1_1` 的 train/test JSON）。可用官方仓库生成后拷入，或 `--data_root` 指定路径。

---

## 2. 一键复现实验命令

```bash
bash run_all.sh
```

- 跑齐 12 组实验（4 数据集 × FedAvg/FedProx + Figure 1 的 4 组 straggler 实验）
- 生成汇总表与全部图表到 `output/`，日志到 `logs/`

仅基础 8 组 + 基础图：`bash run_all_synthetic.sh`。多卡并行：`bash run_all_parallel.sh`。

---

## 3. 关键结果（图表）

跑完 `run_all.sh` 后，`output/` 下包含：

| 图表 | 文件 | 说明 |
|------|------|------|
| accuracy vs rounds | `accuracy_vs_rounds.pdf` | 4 数据集 × FedAvg/FedProx，test accuracy |
| loss vs rounds | `loss_vs_rounds.pdf` | 同上，test loss |
| 论文 Figure 1 | `figure1_stragglers.pdf` | 系统异构：0%/50%/90% stragglers，training loss |
| 论文 Figure 2 | `figure2_data_heterogeneity.pdf` | 数据异构：training loss + dissimilarity |

**关键结果**：汇总数值见 `output/summary.csv`（如 FedProx µ=1 在 non-IID 上 final accuracy 高于 FedAvg）。

---

## 4. 关键超参数与随机种子

| 超参数 | 默认值 |
|--------|--------|
| num_rounds | 200 |
| clients_per_round | 10 |
| num_epochs | 20 |
| batch_size | 10 |
| learning_rate | 0.01 |
| mu（FedProx） | 1.0；FedAvg 等价 mu=0 |
| seed | 0 |

复现请保持默认 `--seed 0`。多 seed 可选：`python run_multi_seed.py --seeds 0,1,2`（生成 mean±std 图）。

---

## 5. 实验日志与原始输出

- **训练/评估日志**：`logs/<name>.jsonl`、`logs/<name>.csv`（每轮 round, accuracy, loss, train_loss, dissimilarity）
- **配置快照**：`output/<name>_config.json`（当次运行参数，可追溯）
- **结果与曲线数据**：`output/<name>_metrics.json`（含 history）；`output/summary.csv`、`summary.json`（汇总最终指标）

---

## 6. 论文与仓库

- 论文：Li T, Sahu A K, Zaheer M, et al. Federated Optimization in Heterogeneous Networks. MLSys 2020.
- 代码仓库：见提交说明中的 GitHub（或等价）链接。

**与论文差异**：本复现采用 200 轮、每轮 10 客户端、20 本地 epoch，数据与模型（60 维 synthetic + MCLR）与 FedProx 原仓一致。
