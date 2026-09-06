# UzbekPlovNER

Воспроизводимый проект multilingual NER для узбекских, русских, английских и
смешанных текстов. Поддерживаются классы `ORG`, `NAME`, `GEO` и точные
символьные интервалы Unicode `[start, end)`.

## Требования и установка

Поддерживается Python `>=3.11,<3.13`. Зависимости устанавливаются через `uv`:

```bash
uv sync --extra dev
uv run python main.py --help
```

## Данные

Датасеты не входят в репозиторий. Пути к локальным файлам задаются в
`exps/<experiment>/configs/default.yaml` или через CLI:

```yaml
paths:
  train: data/my_dataset/train.jsonl
  dev: data/my_dataset/dev.jsonl
  inference_input: data/test_inputs.jsonl

inference:
  output: artifacts/res.jsonl
```

Train и dev содержат разметку:

```json
{"hash":"example-001","text":"Ali Toshkentda ishlaydi.","entities":[{"label":"NAME","start":0,"end":3}]}
```

Inference-файл содержит только `hash` и `text`:

```json
{"hash":"example-002","text":"Aziza Samarqandga bordi."}
```

Относительные пути разрешаются от корня проекта. Абсолютные пути также
поддерживаются. Каталог `data/` исключён из Git, данные можно будет скачать по ссылке из google disk.

## Эксперименты

| Эксперимент | Encoder | NER-head | Особенности |
| --- | --- | --- | --- |
| `exp_baseline` | multilingual DistilBERT | BIO + Linear | Базовый token classification, опциональная cluster augmentation |
| `exp_labse_globalpointer` | LaBSE | EfficientGlobalPointer | Прямое предсказание spans, RoPE, опциональная cluster augmentation |
| `exp_xlmr_large_globalpointer` | XLM-RoBERTa-large | GlobalPointer | `max_length=512`, RoPE, cluster augmentation, нормализация узбекских апострофов |

`exp_xlmr_large_globalpointer` рассчитан на объединённый и очищенный обучающий
датасет из трёх источников. Его реализация находится полностью внутри
`exps/exp_xlmr_large_globalpointer/`.

## Обучение и инференс

Обучение XLM-RoBERTa-large + GlobalPointer:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train
```

Получение итогового JSONL:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage predict \
  --set paths.inference_input=data/test_inputs.jsonl \
  --set inference.output=artifacts/res.jsonl
```

Оценка текущей модели на dev и запуск HTTP-сервиса:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage evaluate
uv run python main.py --exp exp_xlmr_large_globalpointer --stage serve
```

Любой параметр YAML можно переопределить через повторяемый `--set`:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train \
  --set training.batch_size=4 \
  --set training.gradient_accumulation_steps=4 \
  --set augmentation.probability=0.2
```

Приоритет настроек: defaults < YAML < переменные окружения < CLI. Конфигурация
валидируется через Pydantic; `.env` читается через `pydantic-settings`.

## Cluster-based augmentation

Аугментация применяется только к train. Поверхности сущностей каждого класса
векторизуются через input embeddings encoder и кластеризуются отдельно для
`ORG`, `NAME` и `GEO`. Для выбранной записи сущность заменяется близким
кандидатом того же класса. После замены offsets пересчитываются для всех
сущностей, включая незаменённые.

```yaml
augmentation:
  enabled: true
  probability: 0.15
  num_clusters: 16
  max_candidates: 3
```

`probability` — вероятность создать одну аугментированную копию train-записи.
При `enabled: false` или `probability: 0.0` исходный train не изменяется.

## Артефакты и resume

Каждая стадия `train`, `predict` или `evaluate` создаёт отдельный каталог в
`exps/<experiment>/artifacts/runs/` с resolved-конфигурацией и метаданными.

Стадия `train` сохраняет:

- `model/` — лучшую по dev-метрике модель и tokenizer;
- `checkpoint.pt` — состояние для продолжения обучения;
- `metrics_history.json` и `training_summary.json`;
- `prediction.jsonl` и `dev_metrics.json` для лучшей модели.

После обучения `exps/<experiment>/artifacts/model/` обновляется лучшей моделью
запуска. Именно этот каталог используют `predict` и `serve`. Для resume:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage train \
  --set training.resume=exps/exp_xlmr_large_globalpointer/artifacts/runs/<run>/checkpoint.pt \
  --set training.epochs=10
```

Артефакты и веса моделей исключены из Git.

## HTTP API

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage serve
python scripts/check_service.py --url http://localhost:8000
```

Обязательные endpoints:

- `GET /healthz`;
- `POST /api/v1/predict`.

Запрос `/api/v1/predict` — непустой JSON-массив объектов `hash` и `text`.
Ответ содержит исходный `hash` и массив exact spans:

```json
{"data":[{"hash":"example-002","entities":[{"label":"NAME","start":0,"end":5}]}]}
```

Дополнительный `POST /api/v1/predict/normalized` сохраняет исходные offsets и
добавляет к сущности её исходный текст и каноническое значение:

```json
{"data":[{"hash":"example-003","entities":[{"label":"GEO","start":0,"end":10,"text":"Toshkentda","normalized":"Toshkent"}]}]}
```

Канонические соответствия задаются в YAML:

```yaml
normalization:
  aliases:
    GEO:
      Toshkentda: Toshkent
      г. Ташкент: Ташкент
    ORG:
      ООО "Ромашка": Ромашка
```

Поиск aliases регистронезависимый. Если alias отсутствует, выполняется только
безопасная нормализация Unicode, пробелов и вариантов апострофов.

## Docker

Сначала обучите модель выбранного эксперимента. Затем соберите и запустите
образ:

```bash
docker build --build-arg DEFAULT_EXP=exp_xlmr_large_globalpointer -t ner-uz-solution .
docker run --rm -p 8000:8000 ner-uz-solution
```

Все веса и tokenizer должны находиться внутри образа. Во время запуска
контейнер не скачивает модели, не обращается к внешним API и не требует `.env`.
Активный эксперимент можно выбрать через `EXP_NAME`; образ при этом имеет
заданный во время сборки default-эксперимент.

## Добавление эксперимента

Создайте `exps/<name>/` с `main.py` и `configs/default.yaml`. В `main.py`
реализуйте `train`, `predict`, `evaluate` и `serve` с сигнатурой
`(config, context)`. Общие конфигурация, контракты, артефакты и HTTP-адаптер
находятся в `src/ner_core/`.
