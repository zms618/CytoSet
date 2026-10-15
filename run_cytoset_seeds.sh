#!/usr/bin/env bash
set -euo pipefail
# Activate your Python environment before running this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for seed in 42 2026 3407; do
  mkdir -p "output/rtdetr_cytoset_seed${seed}"
  echo "Starting CytoSet seed ${seed} at $(date)"
  python tools/train.py \
    -c "configs/rtdetr/rtdetr_cytoset_seed${seed}.yml" \
    --seed "${seed}" \
    > "output/rtdetr_cytoset_seed${seed}/train_console.log" 2>&1
  echo "Completed CytoSet seed ${seed} at $(date)"
done
