# exp_baseline

Official Transformer baseline implemented as a self-contained experiment.

## Commands

```bash
uv run python main.py --exp exp_baseline --stage train
uv run python main.py --exp exp_baseline --stage predict
uv run python main.py --exp exp_baseline --stage evaluate
uv run python main.py --exp exp_baseline --stage serve
```

Override YAML values from the command line:

```bash
uv run python main.py --exp exp_baseline --stage train \
  --set training.epochs=5 \
  --set training.learning_rate=2e-5
```

Training writes a complete run under `artifacts/runs/` and automatically updates
the current model in `artifacts/model/`. The current model is used by `predict`
and `serve` and is included in the Docker image.

After every epoch, training prints and stores aggregated exact-span metrics for
both `train` and `val`: per-label, micro, and macro precision/recall/F1 with
TP/FP/FN counts. The history is saved in `training_summary.json` and
`metrics_history.json`.

## Cluster-based augmentation

The baseline supports optional label-preserving cluster augmentation. It builds
label-specific K-Means clusters from the base model's token embeddings and
replaces an entity with a similar entity from the same label and cluster. New
spans are recalculated in Unicode character offsets.

The feature is disabled by default. Enable it in `configs/default.yaml`:

```yaml
augmentation:
  enabled: true
  probability: 0.25
  num_clusters: 8
  max_candidates: 5
```

The probability is applied independently to each train record. The dev split is
never augmented. When variants are generated, the complete training data used
by the run is saved as `artifacts/runs/<run>/augmented_train.jsonl`, and the
configuration and counters are saved in `training_summary.json`.

The same settings can be overridden from the CLI:

```bash
uv run python main.py --exp exp_baseline --stage train \
  --set augmentation.enabled=true \
  --set augmentation.probability=0.25
```
