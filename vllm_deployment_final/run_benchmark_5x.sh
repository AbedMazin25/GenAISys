#!/bin/bash
# Run vllm_benchmark.py 5 times sequentially, saving all output to a timestamped log file.

cd "$(dirname "$0")"
source /home/ubuntu/takeaway/venv/bin/activate

LOGFILE="benchmark_5x_$(date +%Y%m%d_%H%M%S).log"

echo "Running benchmark 5 times. Output: $LOGFILE"

export PYTHONUNBUFFERED=1

for i in 1 2 3 4 5; do
    echo "===== RUN $i / 5 — $(date) =====" | tee -a "$LOGFILE"
    python vllm_benchmark.py 2>&1 | tee -a "$LOGFILE"
done

echo "All 5 runs complete. Log saved to $LOGFILE"
