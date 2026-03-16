#!/bin/bash
# Run experiments for Figure 1 (systems heterogeneity: 0%, 50%, 90% stragglers).
# One dataset (synthetic_1_1), drop_percent in {0, 0.5, 0.9}, mu in {0, 1}.
# Then run: python plot_figure1.py --dataset synthetic_1_1 --output_dir output
set -e
OUTPUT_DIR="${OUTPUT_DIR:-output}"
LOG_DIR="${LOG_DIR:-logs}"
DATASET="${DATASET:-synthetic_1_1}"
ROUNDS="${NUM_ROUNDS:-200}"

for DROP in 0 0.5 0.9; do
  echo "=== drop_percent=$DROP mu=0 ==="
  python run_synthetic.py --dataset "$DATASET" --optimizer fedavg --mu 0 --drop_percent "$DROP" --num_rounds "$ROUNDS" --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR"
  echo "=== drop_percent=$DROP mu=1 ==="
  python run_synthetic.py --dataset "$DATASET" --optimizer fedprox --mu 1 --drop_percent "$DROP" --num_rounds "$ROUNDS" --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR"
done
echo "Done. Plot: python plot_figure1.py --dataset $DATASET --output_dir $OUTPUT_DIR"
