for mode in sequence graph combined
do
  python3 train_dag_delta.py \
    data/dag_1k.jsonl \
    --feature-mode "$mode" \
    --horizon 3 \
    --history-length 16 \
    --min-edge-confidence 0.6 \
    --model-out "checkpoints/dag_${mode}_h3.cbm" \
    --report-out "reports/dag_${mode}_h3.json"
done