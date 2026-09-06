# XLM-RoBERTa-large + GlobalPointer

Эксперимент совместно дообучает полный encoder `FacebookAI/xlm-roberta-large` и
GlobalPointer с RoPE для прямого предсказания spans `(label, start, end)`.
BIO/BILOU и CRF не используются.

Перед токенизацией train, dev и inference выполняется сохраняющая длину
нормализация вариантов апострофов в узбекских `Oʻ/Gʻ`. Cluster augmentation
опционально заменяет сущности кандидатами того же класса, пересчитывает offsets
и сохраняет все незаменённые сущности.

Пути к отсутствующим в репозитории данным задаются в `configs/default.yaml`:

```yaml
paths:
  train: data/merged_mendeley_uzner_v2_empty75/train.jsonl
  dev: data/merged_mendeley_uzner_v2_empty75/dev.jsonl
  inference_input: data/test_inputs.jsonl

inference:
  output: artifacts/res.jsonl
```

Запуск из корня проекта:

```powershell
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train
uv run python main.py --exp exp_xlmr_large_globalpointer --stage predict
uv run python main.py --exp exp_xlmr_large_globalpointer --stage evaluate
uv run python main.py --exp exp_xlmr_large_globalpointer --stage serve
```

Default `batch_size=16` воспроизводит исходную конфигурацию. При нехватке VRAM
уменьшите batch и увеличьте gradient accumulation, сохранив effective batch size.
