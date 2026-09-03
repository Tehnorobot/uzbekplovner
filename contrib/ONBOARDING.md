# Onboarding

## Требования

- Python 3.11 или новее;
- `uv` 0.8.5 или совместимая версия;
- Docker для проверки контейнера;
- CUDA 12.4 и NVIDIA Container Toolkit, если используется GPU.

Проект можно открыть через VS Code Dev Containers. После создания контейнера
зависимости устанавливаются автоматически командой `uv sync --extra dev`.

## Локальная установка

Из корня проекта:

```bash
uv sync --extra dev
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS
```

Файл `.env` не коммитится. Модельные веса и tokenizer для обучения могут быть
загружены заранее или указаны локальным путём в конфигурации эксперимента.

## Быстрый запуск

Показать доступные стадии:

```bash
uv run python main.py --help
```

Запустить обучение:

```bash
uv run python main.py --exp exp_baseline --stage train
```

Запустить инференс и оценку текущей модели:

```bash
uv run python main.py --exp exp_baseline --stage predict
uv run python main.py --exp exp_baseline --stage evaluate
```

Обучение также сохраняет модель конкретного запуска в
`exps/exp_baseline/artifacts/runs/<run>/model`, а актуальную модель — в
`exps/exp_baseline/artifacts/model/`.

```bash
uv run python main.py --exp exp_baseline --stage serve
```

## Конфигурация

Конфигурация эксперимента находится в
`exps/<experiment>/configs/default.yaml`. Приоритет значений:

1. значения по умолчанию в коде;
2. YAML;
3. переменные окружения;
4. CLI `--set KEY=VALUE`.

Вложенные переменные `.env` используют двойное подчёркивание, например
`SERVICE__PORT=8000` и `SERVICE__DEVICE=cuda`.

Пример:

```bash
uv run python main.py --exp exp_baseline --stage train \
  --set training.epochs=5 \
  --set training.learning_rate=2e-5 \
  --set training.seed=42
```

Для возобновления используется checkpoint с состоянием модели, optimizer,
scheduler и генераторов случайных чисел:

```bash
uv run python main.py --exp exp_baseline --stage train \
  --set training.resume=exps/exp_baseline/artifacts/runs/<run>/checkpoint.pt \
  --set training.epochs=5
```

## HTTP-проверка

Сервис обязан отвечать на `GET /healthz` и `POST /api/v1/predict`:

```bash
python scripts/check_service.py --url http://localhost:8000
python scripts/evaluate_service.py \
  --url http://localhost:8000 \
  --gold data/dev.jsonl \
  --predictions exps/exp_baseline/artifacts/service/dev_predictions.jsonl \
  --output exps/exp_baseline/artifacts/service/dev_metrics.json
```

Перед сохранением ответы проходят общую проверку exact-span контракта: UTF-8,
Unicode offsets, допустимые labels, порядок hash, дубликаты и пересечения.

## Docker

Docker-образ собирается после появления локальной модели эксперимента:

```bash
docker build --build-arg DEFAULT_EXP=exp_baseline -t ner-uz-solution .
docker run --rm -p 8000:8000 ner-uz-solution
# Для GPU добавьте к docker run: --gpus all
```

`HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1` гарантируют отсутствие скачивания
моделей после запуска контейнера. Реальный `.env` и папка `ТЗ` в образ не
попадают.

## Проверки качества кода

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
