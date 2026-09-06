# LaBSE + EfficientGlobalPointer

This experiment fine-tunes the complete `sentence-transformers/LaBSE` encoder jointly
with an EfficientGlobalPointer span head. It predicts `(label, start, end)` spans directly;
BIO/BILOU tags and CRF decoding are not used.

Cluster-based entity replacement is optional. It only replaces entities with candidates
from the same label, recalculates offsets, and retains every entity that was not replaced.

## Run locally

Install dependencies from the repository root:

```powershell
uv sync --extra dev
```

Train with the default YAML:

```powershell
uv run python main.py --exp exp_labse_globalpointer --stage train
```

Disable augmentation or override any other YAML value from the CLI:

```powershell
uv run python main.py --exp exp_labse_globalpointer --stage train `
  --set augmentation.enabled=false `
  --set training.batch_size=4
```

After every epoch, train/dev exact-span metrics are printed and appended to
`metrics_history.json`. The run also contains `prediction.jsonl`, `dev_metrics.json`,
`checkpoint.pt`, `training_summary.json`, and the local-only `model/` bundle.

Run prediction, evaluation, or the HTTP service using the current model:

```powershell
uv run python main.py --exp exp_labse_globalpointer --stage predict
uv run python main.py --exp exp_labse_globalpointer --stage evaluate
uv run python main.py --exp exp_labse_globalpointer --stage serve
```

The default merged train set is large. For a quick smoke run, point `paths.train` and
`paths.dev` at small valid JSONL files through `--set`.
