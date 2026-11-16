#!/usr/bin/env bash
# Train one shard of LSTM-AE jobs and push the saved errors to GitHub.
# Usage (inside the cloned repo):  bash aws/run_shard.sh <shard> <num_shards>
set -uo pipefail
SHARD=${1:?shard}; SHARDS=${2:?num_shards}
cd "$(dirname "$0")/.."
mkdir -p logs results/scores
LOG=logs/shard_${SHARD}_of_${SHARDS}.log

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q numpy pandas scipy scikit-learn
fi

push_results () {
  git add results/scores logs >/dev/null 2>&1
  git commit -q -m "AWS shard ${SHARD}/${SHARDS}: $(ls results/scores | wc -l) score files" || return 0
  for i in 1 2 3 4 5; do
    git pull -q --rebase origin main && git push -q origin HEAD:main && return 0
    sleep $((RANDOM % 20 + 5))
  done
}

.venv/bin/python scripts/train_score.py --shard "$SHARD" --of "$SHARDS" >> "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 900            # push progress every 15 minutes
  push_results
done
wait $PID; echo "exit code $?" >> "$LOG"
push_results
