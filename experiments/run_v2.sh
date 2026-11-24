#!/usr/bin/env bash
# GNN v2 end to end: two rounds of (labels -> 5 trainings -> checkpoint selection),
# then the final test of the round-2 ensemble on the held-out test instances.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
R=experiments/results/v2
mkdir -p $R
ens() { $PY -c "import json,sys; d=json.load(open('$R/$1_selected.json')); print('ens:'+','.join(v['ckpt'] for v in d.values()))"; }

POL0="ens:$(ls experiments/results/retrain/sage_seed[0-4].pt | tr '\n' ',' | sed 's/,$//')"
for round in r1 r2; do
  if [ $round = r1 ]; then POL=$POL0; else POL=$(ens r1); fi
  echo "== $round labels (policy $POL)"
  $PY experiments/gnn_v2_labels.py --policy "$POL" --out $R/labels_$round.pt --workers 8
  echo "== $round training"
  for s in 0 1 2 3 4; do $PY experiments/gnn_v2_train.py --labels $R/labels_$round.pt --seed $s --tag $round > $R/train_${round}_seed$s.log 2>&1 & done
  wait
  echo "== $round selection"
  $PY experiments/gnn_v2_select.py --tag $round --workers 8
done
echo "== final test"
$PY experiments/rev_runner.py v2final --workers 8
echo "V2_DONE"
