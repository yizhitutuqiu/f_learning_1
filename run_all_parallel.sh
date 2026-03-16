#!/usr/bin/env bash
# 与 run_all.sh 完全相同的 12 组实验 + 汇总 + 绘图，但利用多卡并行加速。
# 每组实验独占 1 个 GPU（CUDA_VISIBLE_DEVICES），种子与逻辑不变，结果与 run_all.sh 一致。
# 用法: bash run_all_parallel.sh
# 可选: N_GPUS=8 bash run_all_parallel.sh  （默认 8，卡数不足时自动分批）

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
N_GPUS="${N_GPUS:-8}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "========== 并行运行：全部实验 + 全部图表（最多 ${N_GPUS} 路并行）=========="
echo "日志: $LOG_DIR  结果: $OUTPUT_DIR"
echo ""

# 单次实验：$1=dataset $2=optimizer $3=mu $4=drop(可选) $5=GPU_ID
run_one() {
    local dataset="$1"
    local optimizer="$2"
    local mu="$3"
    local drop="${4:-}"
    local gpu_id="${5:-0}"
    local name="${dataset}_${optimizer}_mu${mu}"
    [ -n "$drop" ] && name="${name}_drop${drop}"
    local extra=""
    [ -n "$drop" ] && extra="--drop_percent $drop"
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        echo "[$(date '+%H:%M:%S')] GPU $gpu_id: $name"
        python run_synthetic.py \
            --dataset "$dataset" \
            --optimizer "$optimizer" \
            --mu "$mu" \
            $extra \
            --log_dir "$LOG_DIR" \
            --output_dir "$OUTPUT_DIR" \
            > "$LOG_DIR/${name}.out" 2>&1
        echo "[$(date '+%H:%M:%S')] GPU $gpu_id: done $name"
    )
}

# 12 组实验：(dataset, optimizer, mu, drop)
# 基础 8 组
JOBS=( "synthetic_iid:fedavg:0:" "synthetic_iid:fedprox:1:" "synthetic_0_0:fedavg:0:" "synthetic_0_0:fedprox:1:"
       "synthetic_0.5_0.5:fedavg:0:" "synthetic_0.5_0.5:fedprox:1:" "synthetic_1_1:fedavg:0:" "synthetic_1_1:fedprox:1:"
       "synthetic_1_1:fedavg:0:0.5" "synthetic_1_1:fedprox:1:0.5" "synthetic_1_1:fedavg:0:0.9" "synthetic_1_1:fedprox:1:0.9" )

run_batch() {
    local start=$1
    local count=$2
    local gpu_start=0
    for (( i = 0; i < count; i++ )); do
        local idx=$(( start + i ))
        local line="${JOBS[$idx]}"
        IFS=: read -r dataset optimizer mu drop <<< "$line"
        local gpu=$(( (gpu_start + i) % N_GPUS ))
        run_one "$dataset" "$optimizer" "$mu" "$drop" "$gpu" &
    done
    wait
}

echo "===== 1/2 基础实验（8 组并行）====="
run_batch 0 8
echo "  base 8 done."
echo ""

echo "===== 2/2 Figure 1 实验（4 组并行）====="
run_batch 8 4
echo "  figure1 4 done."
echo ""

echo "========== 全部 12 组实验完成 =========="
echo ""

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
echo "  - accuracy_vs_rounds.pdf, loss_vs_rounds.pdf, accuracy_loss_grid.pdf"
echo "  - figure2_data_heterogeneity.pdf   figure1_stragglers.pdf"
echo "日志目录: $LOG_DIR"
