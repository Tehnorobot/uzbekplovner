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
Датасеты также не хранятся в репозитории: локальные пути к train, dev и
inference JSONL задаются в YAML или через CLI.

## Быстрый запуск

Показать доступные стадии:

```bash
uv run python main.py --help
```

Запустить обучение:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train
```

После каждой эпохи обучение выводит метрики для `train` и `val`: loss,
exact-span precision/recall/F1, TP/FP/FN, micro и macro агрегаты. История
сохраняется в
`exps/exp_xlmr_large_globalpointer/artifacts/runs/<run>/metrics_history.json`.

Запустить инференс и оценку текущей модели:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage predict
uv run python main.py --exp exp_xlmr_large_globalpointer --stage evaluate
```

Обучение также сохраняет модель конкретного запуска в
`exps/exp_xlmr_large_globalpointer/artifacts/runs/<run>/model`, а актуальную
модель — в `exps/exp_xlmr_large_globalpointer/artifacts/model/`.

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage serve
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
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train \
  --set training.epochs=5 \
  --set training.learning_rate=2e-5 \
  --set augmentation.probability=0.2
```

Для `exp_xlmr_large_globalpointer` cluster augmentation создаёт дополнительные
train-записи, заменяя сущности кандидатами того же класса. После замены offsets
пересчитываются для всех сущностей. Аугментация управляется полями `enabled`,
`probability`, `num_clusters` и `max_candidates` в секции `augmentation`.

Для возобновления используется checkpoint с состоянием модели, optimizer,
scheduler и генераторов случайных чисел:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train \
  --set training.resume=exps/exp_xlmr_large_globalpointer/artifacts/runs/<run>/checkpoint.pt \
  --set training.epochs=5
```

## HTTP-проверка

Сервис отвечает на `GET /healthz`, обязательный `POST /api/v1/predict` и
дополнительный `POST /api/v1/predict/normalized`. Последняя ручка сохраняет
исходные offsets и добавляет к каждой сущности поля `text` и `normalized`.

Быстрая проверка обязательной ручки:

```bash
python scripts/check_service.py --url http://localhost:8000
```

Перед сохранением ответы проходят общую проверку exact-span контракта: UTF-8,
Unicode offsets, допустимые labels, порядок hash, дубликаты и пересечения.

## Docker

Docker-образ собирается после появления локальной модели эксперимента:

```bash
docker build --build-arg DEFAULT_EXP=exp_xlmr_large_globalpointer -t ner-uz-solution .
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
