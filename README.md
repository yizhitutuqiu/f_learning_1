# FedProx 复现：Flower + PyTorch

基于 **Flower** 与 **PyTorch** 在单机仿真下复现论文 [Federated Optimization in Heterogeneous Networks (FedProx)](https://arxiv.org/abs/1812.06127)（MLSys 2020）的 synthetic 实验，并与 FedAvg 对比。

---

## 1. 环境安装

**依赖**：Python 3.8+，见 `requirements.txt`。

```bash
cd f_learning
pip install -r requirements.txt
```

| 依赖 | 说明 |
|------|------|
| flwr | Flower 联邦学习框架 |
| torch | PyTorch |
| numpy>=1.21,<2 | 与 PyTorch 兼容 |
| matplotlib | 画图 |
| tqdm | 进度条 |

**数据**：当前代码从 **与 `f_learning` 同级的 `FedProx` 目录** 加载（即 `dataset.py` 中 `FEDPROX_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "FedProx", "data")`），实际路径为：

- **默认加载路径**：`Federated_learning_1/FedProx/data/`
- 其下需有子目录：`synthetic_iid/data/{train,test}/`、`synthetic_0_0/`、`synthetic_0.5_0.5/`、`synthetic_1_1/`，每个里含 `data/train/*.json`、`data/test/*.json`（如 `mytrain.json`）。

若数据放在别处，可通过 `--data_root /path/to/FedProx/data` 指定。数据可从 [FedProx 官方仓库](https://github.com/litian96/FedProx) 克隆后，在其 `data/synthetic_*/` 下按 README 生成 JSON，再保证目录结构与上述一致（或把生成好的 `data/` 拷到 `Federated_learning_1/FedProx/data/`）。

---

## 2. 一键复现实验命令

### 一次脚本跑完所有事（推荐）

**一条命令**跑齐全部实验并生成**所有图表**（含论文 Figure 1 / Figure 2 与原有 accuracy、loss 图）：

```bash
cd f_learning
bash run_all.sh
```

该脚本会：

1. 运行 **12 组实验**：8 组基础（4 数据集 × FedAvg/FedProx，无 straggler）+ 4 组 Figure 1（synthetic_1_1 × 50%/90% stragglers × µ=0/1；0% 复用基础实验）
2. 生成汇总表：`output/summary.csv`、`output/summary.json`
3. 生成**全部图表**：
   - 原有图：`accuracy_vs_rounds.pdf`、`loss_vs_rounds.pdf`、`accuracy_loss_grid.pdf`
   - 论文 Figure 2：`output/figure2_data_heterogeneity.pdf`（数据异构：training loss + dissimilarity）
   - 论文 Figure 1：`output/figure1_stragglers.pdf`（系统异构：0%/50%/90% stragglers）

结果与日志分别写入 `output/`、`logs/`。

### 仅跑基础 8 组 + 基础图（不跑 Figure 1 实验）

若只需 4 数据集 × FedAvg/FedProx 及对应的 accuracy/loss 图（不画 Figure 1）：

```bash
bash run_all_synthetic.sh
```

该脚本会：

- 依次运行 8 组实验（synthetic_iid / synthetic_0_0 / synthetic_0.5_0.5 / synthetic_1_1 × FedAvg / FedProx）
- 将**实验结果**写入 `output/`，**日志**写入 `logs/`
- 最后调用 `summarize_results.py` 与 `plot_results.py`，在 `output/` 下生成汇总表与**基础图表**（无 Figure 1/2）

**单次单实验**（例如只跑 synthetic_1_1 + FedProx）：

```bash
python run_synthetic.py --dataset synthetic_1_1 --optimizer fedprox --mu 1 --log_dir logs --output_dir output
```

**多随机种子 + 均值±标准差图**（可选）：

```bash
python run_multi_seed.py --dataset synthetic_1_1 --optimizer fedprox --mu 1 --seeds 0,1,2
# 输出：output/accuracy_vs_rounds_mean_std.pdf, output/loss_vs_rounds_mean_std.pdf
```

---

## 3. 关键超参数与随机种子

与 FedProx 论文 / 原仓设定对齐的主要超参数如下（可在 `run_synthetic.py` 中通过命令行覆盖）：

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `--num_rounds` | 200 | 联邦轮数 |
| `--clients_per_round` | 10 | 每轮参与客户端数 |
| `--num_epochs` | 20 | 客户端本地训练 epoch |
| `--batch_size` | 10 | 本地 batch 大小 |
| `--learning_rate` | 0.01 | 学习率 |
| `--mu` | 1.0（FedProx）/ 0（FedAvg） | 近端项系数 |
| `--drop_percent` | 0.0 | 系统异构：被选为 straggler 的比例（0 / 0.5 / 0.9），straggler 本地 epoch 为 1～num_epochs-1 |
| `--seed` | 0 | 随机种子（控制数据划分与采样等） |

复现时建议固定 **`--seed 0`**（默认）以便结果可复现；做多 seed 时使用 `run_multi_seed.py --seeds 0,1,2` 等。

---

## 4. 实验日志与原始输出（可追溯）

每次运行会在 **`logs/`** 与 **`output/`** 下产生以下文件（文件名含数据集与算法，例如 `synthetic_1_1_client10_epoch20_mu1`）：

- **训练/评估日志**
  - `logs/<name>.jsonl`：每行一个 round，字段 `round`, `accuracy`, `loss`，以及（若有）`train_loss`, `dissimilarity`
  - `logs/<name>.csv`：表头 `round,accuracy,loss,train_loss,dissimilarity`，便于做表或后续画图
  - `logs/<name>.txt`：与 FedProx 原版 plot 脚本兼容的文本日志
- **每次实验的配置文件快照**
  - `output/<name>_config.json`：当次运行的全部命令行参数与解析后的 `dataset_resolved`，保证可追溯
- **结果与曲线数据**
  - `output/<name>_metrics.json`：含 `final_accuracy`、`final_loss` 及每轮 `history`（含 `train_loss`、`dissimilarity`，用于 Figure 2）
  - `output/summary.csv`、`output/summary.json`：由 `summarize_results.py` 在跑完 `run_all_synthetic.sh` 后生成，汇总所有实验的最终指标

---

## 5. 图表

脚本 `plot_results.py` 会生成 2 张与论文对应的图（均在 **`output/`** 下）：

1. **Test accuracy vs rounds**  
   - 文件：`output/accuracy_vs_rounds.pdf`（以及 `accuracy.pdf`）  
   - 内容：4 个 synthetic 数据集上 FedAvg vs FedProx 的 **accuracy vs rounds** 曲线（每数据集一子图或合并图，见脚本逻辑）。

2. **Test loss vs rounds**  
   - 文件：`output/loss_vs_rounds.pdf`（以及 `loss.pdf`）  
   - 内容：同上，**loss vs rounds**。

若使用多随机种子（`run_multi_seed.py`），还会生成：

- `output/accuracy_vs_rounds_mean_std.pdf`
- `output/loss_vs_rounds_mean_std.pdf`  

为**均值曲线 + 标准差阴影**（可选，符合“若做多次随机种子：用均值曲线 + 阴影/误差条”）。

### 论文 Figure 1 / Figure 2 复现

- **Figure 1（系统异构 / stragglers）**：0%、50%、90% 设备为 straggler 时 FedAvg (µ=0) vs FedProx (µ>0) 的 **training loss**。
  1. 对同一数据集（如 `synthetic_1_1`）在 `drop_percent=0, 0.5, 0.9` 下各跑 µ=0 与 µ=1：
     ```bash
     bash run_figure1_stragglers.sh
     ```
  2. 画图：`python plot_figure1.py --dataset synthetic_1_1 --output_dir output`  
  3. 输出：`output/figure1_stragglers.pdf`（3 列：0% / 50% / 90% stragglers，每列两条曲线）。

- **Figure 2（数据异构）**：无 straggler 时，4 个 synthetic 数据集的 **training loss**（上排）与 **dissimilarity（梯度方差）**（下排）。
  1. 先跑齐 4 个数据集 × µ=0、µ=1（如 `bash run_all_synthetic.sh`），保证 `history` 中含 `train_loss` 与 `dissimilarity`（当前 `run_synthetic.py` 已写入）。
  2. 画图：`python plot_figure2.py --output_dir output`  
  3. 输出：`output/figure2_data_heterogeneity.pdf`（上排 4 子图 training loss，下排 4 子图 dissimilarity）。

**你复现的关键结果**：可将上述 `output/` 下的 PDF 截图或插入到报告/README 中，例如：

```markdown
### 关键结果截图/图表
- accuracy vs rounds: 见 `output/accuracy_vs_rounds.pdf`
- loss vs rounds: 见 `output/loss_vs_rounds.pdf`
```


---

## 6. 目录结构概览

```
f_learning/
├── README.md                 # 本说明（可复现说明）
├── requirements.txt         # 环境依赖
├── dataset.py               # 读取 FedProx synthetic JSON
├── model.py                 # MCLR (60→10)
├── client.py                # FedProx 客户端（含近端项）
├── run_synthetic.py         # 单次实验入口
├── run_all.sh               # 一键：12 组实验 + 汇总 + 全部图（含 Figure 1/2）
├── run_all_synthetic.sh     # 仅 8 组基础实验 + 汇总 + 基础图
├── run_multi_seed.py        # 多 seed 运行 + mean±std 图
├── run_figure1_stragglers.sh # 仅 Figure 1 实验：0%/50%/90% stragglers（6 组）
├── plot_results.py          # accuracy/loss vs rounds 图表
├── plot_figure1.py         # 论文 Figure 1：系统异构（stragglers）training loss
├── plot_figure2.py         # 论文 Figure 2：数据异构 training loss + dissimilarity
├── summarize_results.py     # 汇总 output/*_metrics.json → summary.csv/json
├── logs/                    # 实验日志（.jsonl, .csv, .txt）
└── output/                  # 结果、配置快照、图表（.pdf, _config.json, _metrics.json）
```

---

## 7. 论文与代码仓库

- **论文**：Li T, Sahu A K, Zaheer M, et al. Federated Optimization in Heterogeneous Networks. MLSys 2020. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)
- **代码仓库**：请将本目录置于 GitHub（或等价平台）仓库中，并在提交说明中注明仓库链接。
- **论文阅读笔记**：单独提交，不在此仓库内。

---

## 8. 与论文原设定的差异（若做教学缩放）

若你减少了轮数、客户端数或换了更小模型，请在报告/README 中明确写出，例如：

- 本复现使用与论文一致的 **200 轮、每轮 10 客户端、20 本地 epoch**；数据与模型（60 维 synthetic + MCLR）与 FedProx 原仓一致，未做缩放。
- 若你改为 `--num_rounds 50` 等，请注明：“为节省时间将轮数降为 50，结论仅观察趋势，与论文定量结果不可直接对比。”

上述设置满足 mini-project 对 **README（可复现说明）、实验日志与原始输出、图表、关键超参数与随机种子** 的要求。**一键跑全实验并出全图**：**`bash run_all.sh`**；仅基础 8 组+基础图：`bash run_all_synthetic.sh`。
