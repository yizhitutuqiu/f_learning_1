#!/usr/bin/env bash
# 一键完成：全部实验 + 全部图表（含论文 Figure 1 / Figure 2）
# 1) 8 组基础实验（4 数据集 × FedAvg + FedProx，无 straggler）→ 供 plot_results + Figure 2
# 2) 4 组 Figure 1 实验（synthetic_1_1 × 50%/90% stragglers × µ=0/1）；0% 复用步骤 1 的 synthetic_1_1
# 3) 汇总表 + 所有图：accuracy/loss、Figure 1、Figure 2
# 用法: bash run_all.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "========== 一键运行：全部实验 + 全部图表 =========="
echo "日志: $LOG_DIR  结果: $OUTPUT_DIR"
echo ""

run_one() {
    local dataset="$1"
    local optimizer="$2"
    local mu="$3"
    local drop="${4:-}"
    local name="${dataset}_${optimizer}_mu${mu}"
    [ -n "$drop" ] && name="${name}_drop${drop}"
    echo "[$(date '+%H:%M:%S')] Running $name ..."
    local extra=""
    [ -n "$drop" ] && extra="--drop_percent $drop"
    python run_synthetic.py \
        --dataset "$dataset" \
        --optimizer "$optimizer" \
        --mu "$mu" \
        $extra \
        --log_dir "$LOG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/${name}.out"
    echo "[$(date '+%H:%M:%S')] Done $name"
    echo ""
}

# ---------- 1) 基础 8 组（无 straggler），供 accuracy/loss 图与 Figure 2 ----------
echo "===== 1/2 基础实验：4 数据集 × FedAvg + FedProx（无 straggler）====="
run_one "synthetic_iid"     "fedavg"  0
run_one "synthetic_iid"     "fedprox" 1
run_one "synthetic_0_0"     "fedavg"  0
run_one "synthetic_0_0"     "fedprox" 1
run_one "synthetic_0.5_0.5" "fedavg"  0
run_one "synthetic_0.5_0.5" "fedprox" 1
run_one "synthetic_1_1"     "fedavg"  0
run_one "synthetic_1_1"     "fedprox" 1

# ---------- 2) Figure 1 所需：仅 50%、90% stragglers（0% 已由上面 synthetic_1_1 提供）----------
echo "===== 2/2 Figure 1 实验：synthetic_1_1 × 50%/90% stragglers × µ=0/1 ====="
run_one "synthetic_1_1" "fedavg"  0 "0.5"
run_one "synthetic_1_1" "fedprox" 1 "0.5"
run_one "synthetic_1_1" "fedavg"  0 "0.9"
run_one "synthetic_1_1" "fedprox" 1 "0.9"

echo "========== 全部 12 组实验完成 =========="
echo ""

# ---------- 3) 汇总与绘图 ----------
echo "生成汇总表 summary.csv / summary.json ..."
python summarize_results.py --output_dir "$OUTPUT_DIR" 2>/dev/null || true

echo "生成基础曲线图（accuracy vs rounds, loss vs rounds, grid）..."
python plot_results.py --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR" 2>/dev/null || true

echo "生成论文 Figure 2（数据异构：training loss + dissimilarity）..."
python plot_figure2.py --output_dir "$OUTPUT_DIR" 2>/dev/null || true

echo "生成论文 Figure 1（系统异构：stragglers 0%/50%/90%）..."
python plot_figure1.py --dataset synthetic_1_1 --output_dir "$OUTPUT_DIR" 2>/dev/null || true

echo ""
echo "========== 全部完成 =========="
echo "结果目录: $OUTPUT_DIR"
echo "  - summary.csv / summary.json"
echo "  - accuracy_vs_rounds.pdf, loss_vs_rounds.pdf, accuracy_loss_grid.pdf  （原有图）"
echo "  - figure2_data_heterogeneity.pdf   （论文 Figure 2）"
echo "  - figure1_stragglers.pdf            （论文 Figure 1）"
echo "日志目录: $LOG_DIR"
