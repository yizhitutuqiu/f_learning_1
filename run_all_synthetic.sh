#!/usr/bin/env bash
# 复现 FedProx 论文全部 synthetic 实验的 pipeline
# 4 个数据集 × (FedAvg + FedProx) = 8 组实验
# 结果 -> output/，日志 -> logs/
# 用法: bash run_all_synthetic.sh
# 依赖: conda activate fl1 且已 pip install -r requirements.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LOG_DIR="$SCRIPT_DIR/logs"
export OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "========== FedProx 复现 Pipeline =========="
echo "日志目录: $LOG_DIR"
echo "结果目录: $OUTPUT_DIR"
echo ""

run_one() {
    local dataset="$1"
    local optimizer="$2"
    local mu="$3"
    local name="${dataset}_${optimizer}_mu${mu}"
    echo "[$(date '+%H:%M:%S')] Running $name ..."
    python run_synthetic.py \
        --dataset "$dataset" \
        --optimizer "$optimizer" \
        --mu "$mu" \
        --log_dir "$LOG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/${name}.out"
    echo "[$(date '+%H:%M:%S')] Done $name"
    echo ""
}

# 1) synthetic_iid
run_one "synthetic_iid"    "fedavg"  0
run_one "synthetic_iid"    "fedprox" 1

# 2) synthetic_0_0
run_one "synthetic_0_0"    "fedavg"  0
run_one "synthetic_0_0"    "fedprox" 1

# 3) synthetic_0.5_0.5
run_one "synthetic_0.5_0.5" "fedavg"  0
run_one "synthetic_0.5_0.5" "fedprox" 1

# 4) synthetic_1_1
run_one "synthetic_1_1"    "fedavg"  0
run_one "synthetic_1_1"    "fedprox" 1

echo "========== 全部 8 组实验完成 =========="
echo "结果 (metrics): $OUTPUT_DIR/*_metrics.json"
echo "日志 (per-round): $LOG_DIR/*.txt"
echo "运行日志 (stdout): $LOG_DIR/*.out"
echo ""

echo "生成汇总表 summary.csv / summary.json ..."
python summarize_results.py --output_dir "$OUTPUT_DIR" 2>/dev/null || true

echo "生成曲线图 ..."
python plot_results.py --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR" 2>/dev/null || true

echo "========== Pipeline 结束 =========="
echo "最终实验结果: $OUTPUT_DIR/ (summary.csv, summary.json, *_metrics.json, *.pdf)"
echo "日志: $LOG_DIR/ (*.txt, *.out)"
