# f_learning 测试命令

在项目根目录 `Federated_learning_1/f_learning` 下执行。建议使用 conda 环境 `hsmr`。

---

## 1. 环境准备

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning

conda activate fl1
pip install -r requirements.txt
```

**当默认 /tmp 空间不足（No space left on device）时**，改用用户目录下的 tmp 再安装：

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning
conda activate fl1

export TMPDIR=$HOME/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR
pip install -r requirements.txt
```

或直接执行脚本（会设置 TMPDIR 并执行 pip install）：

```bash
conda activate fl1
bash install_with_user_tmp.sh
```

---

## 2. 快速冒烟测试（2 轮，约 1 分钟内）

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning
conda activate hsmr

# FedAvg
python run_synthetic.py --dataset synthetic_iid --optimizer fedavg --mu 0 --num_rounds 2

# FedProx
python run_synthetic.py --dataset synthetic_iid --optimizer fedprox --mu 1 --num_rounds 2
```

成功时会在 `log_synthetic/` 下生成 `*_metrics.json` 和 `*.txt`，并打印 Final test accuracy。

---

## 3. 单次完整实验（200 轮，与 FedProx 论文一致）

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning
conda activate hsmr

# 示例：synthetic_iid 上 FedAvg
python run_synthetic.py --dataset synthetic_iid --optimizer fedavg --mu 0

# 示例：synthetic_iid 上 FedProx (mu=1)
python run_synthetic.py --dataset synthetic_iid --optimizer fedprox --mu 1

# 示例：synthetic_1_1 上 FedProx
python run_synthetic.py --dataset synthetic_1_1 --optimizer fedprox --mu 1
```

---

## 4. 跑齐 4 个数据集 × FedAvg + FedProx（共 8 组，复现论文对比）

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning
conda activate hsmr
bash run_all_synthetic.sh
```

或逐条执行：

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning
conda activate hsmr

python run_synthetic.py --dataset synthetic_iid    --optimizer fedavg  --mu 0
python run_synthetic.py --dataset synthetic_iid    --optimizer fedprox --mu 1
python run_synthetic.py --dataset synthetic_0_0    --optimizer fedavg  --mu 0
python run_synthetic.py --dataset synthetic_0_0    --optimizer fedprox --mu 1
python run_synthetic.py --dataset synthetic_0.5_0.5 --optimizer fedavg  --mu 0
python run_synthetic.py --dataset synthetic_0.5_0.5 --optimizer fedprox --mu 1
python run_synthetic.py --dataset synthetic_1_1    --optimizer fedavg  --mu 0
python run_synthetic.py --dataset synthetic_1_1    --optimizer fedprox --mu 1
```

---

## 5. 生成指标与曲线图

跑完上述实验后：

```bash
cd /data/litengmo/ml-test-1/Federated_learning_1/f_learning
conda activate hsmr
python plot_results.py
```

会生成：

- `loss.pdf`：各数据集上的 loss 曲线（FedAvg vs FedProx）
- `accuracy.pdf`：各数据集上的 test accuracy 曲线
- `accuracy_loss_grid.pdf`：四合一总图

---

## 6. 查看指标结果

- 每轮与最终指标：`log_synthetic/<dataset>_client10_epoch20_mu<0|1>_metrics.json`
- 文本日志（可被 FedProx 的 plot_fig2 解析）：`log_synthetic/<dataset>_client10_epoch20_mu<0|1>.txt`

示例：

```bash
cat log_synthetic/synthetic_iid_client10_epoch20_mu1_metrics.json | head -30
```

---

## 7. 可选参数一览

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | synthetic_iid | synthetic_iid / synthetic_0_0 / synthetic_0.5_0.5 / synthetic_1_1 |
| `--optimizer` | fedprox | fedavg / fedprox |
| `--mu` | 1.0 | 近端项系数，FedAvg 用 0 |
| `--num_rounds` | 200 | 联邦轮数 |
| `--clients_per_round` | 10 | 每轮参与客户端数 |
| `--num_epochs` | 20 | 本地训练 epoch |
| `--batch_size` | 10 | 本地 batch 大小 |
| `--learning_rate` | 0.01 | 学习率 |
| `--log_dir` | log_synthetic | 日志目录 |
| `--data_root` | ../FedProx/data | FedProx 数据根目录 |

示例：自定义轮数与学习率：

```bash
python run_synthetic.py --dataset synthetic_1_1 --optimizer fedprox --mu 1 --num_rounds 100 --learning_rate 0.005
```
