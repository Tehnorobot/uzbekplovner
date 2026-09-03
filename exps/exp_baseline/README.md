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
