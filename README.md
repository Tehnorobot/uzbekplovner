# NER experiments

Воспроизводимый проект NER для узбекских, русских, английских и смешанных
текстов.

## Быстрый старт

Требуется Python 3.11+ и [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run python main.py --help
```

Обучение, предикт и оценка:

```bash
uv run python main.py --exp exp_baseline --stage train
uv run python main.py --exp exp_baseline --stage predict
uv run python main.py --exp exp_baseline --stage evaluate
```

Каждый запуск создаёт каталог в `exps/<experiment>/artifacts/runs/` и сохраняет
`resolved_config.yaml`, `metadata.json`, checkpoint, модель, предсказания и
метрики. Параметры YAML можно переопределять через CLI:

```bash
uv run python main.py --exp exp_baseline --stage train \
  --set training.epochs=5 \
  --set training.learning_rate=2e-5 \
  --set training.seed=42
```

Приоритет настроек: defaults < YAML < переменные окружения < CLI. Итоговая
схема конфигурации валидируется через Pydantic; `.env` читается через
`pydantic-settings`. Вложенные переменные используют `__`, например
`SERVICE__PORT=8000`.

## Возобновление и текущая модель

После обучения checkpoint находится в `<run>/checkpoint.pt`. Для продолжения
обучения укажите его в YAML или CLI:

```bash
uv run python main.py --exp exp_baseline --stage train \
  --set training.resume=exps/exp_baseline/artifacts/runs/<run>/checkpoint.pt \
  --set training.epochs=5
```

После успешного обучения модель конкретного запуска сохраняется в
`artifacts/runs/<run>/model`, а актуальная модель эксперимента автоматически
обновляется в `artifacts/model`. Именно из `artifacts/model` работают `predict`
и `serve`.

## HTTP-сервис

```bash
uv run python main.py --exp exp_baseline --stage serve
python scripts/check_service.py --url http://localhost:8000
python scripts/evaluate_service.py \
  --url http://localhost:8000 \
  --gold data/dev.jsonl \
  --predictions exps/exp_baseline/artifacts/service/dev_predictions.jsonl \
  --output exps/exp_baseline/artifacts/service/dev_metrics.json
```

Обязательные endpoints: `GET /healthz` и `POST /api/v1/predict`. Модель и
tokenizer загружаются с `local_files_only=True`; в контейнере дополнительно
включён offline-режим Transformers/Hugging Face.

## Docker

Сначала нужно обучить модель эксперимента. Затем:

```bash
docker build --build-arg DEFAULT_EXP=exp_baseline -t ner-uz-solution .
docker run --rm -p 8000:8000 ner-uz-solution
# Для GPU добавьте: --gpus all
```

Если в образ упакованы несколько экспериментов, активный выбирается при
запуске:

```bash
docker run --rm -e EXP_NAME=exp_baseline -p 8000:8000 ner-uz-solution
```

Во время запуска контейнер не скачивает модели, не вызывает внешние API и не
требует `.env`. Реальный `.env` используется только локально и не копируется
в образ.

## Добавление эксперимента

Создайте `exps/<name>/` с `main.py` и `configs/default.yaml`. В `main.py`
реализуйте функции `train`, `predict`, `evaluate` и `serve` с
сигнатурой `(config, context)`. Общая валидация контрактов и HTTP-адаптер
находятся в `src/ner_core/`.
