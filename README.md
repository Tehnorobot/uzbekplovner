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
поддерживаются. Каталог `data/` исключён из Git; ссылка на архив с данными
публикуется отдельно.

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

## Использование готовых весов

Ниже приведён полный сценарий для эксперимента
`exp_xlmr_large_globalpointer` (лучший подход по метрике F1 micro). Все команды выполняются из корня репозитория.

### 1. Установка зависимостей

Windows PowerShell:

```powershell
uv sync
```

Linux:

```bash
uv sync
```

### 2. Загрузка и размещение весов

Скачайте ZIP-архив с обученными весами вручную из
[папки Google Drive](https://drive.google.com/drive/folders/1RasFIX0KVxZ42TJG7c1CjCjbj95p8HNb?usp=sharing).
Затем самостоятельно распакуйте его средствами операционной системы или
архиватора в каталог:

```text
exps/exp_xlmr_large_globalpointer/artifacts/model/
```

В этом каталоге должна получиться следующая структура:

```text
model_config.json
pointer_state.pt
encoder/
  config.json
  model.safetensors
tokenizer/
  tokenizer_config.json
  tokenizer.json
  ...
```

### 3. Запуск HTTP-сервера

Windows PowerShell:

```powershell
uv run python main.py --exp exp_xlmr_large_globalpointer --stage serve
```

Linux:

```bash
uv run python main.py --exp exp_xlmr_large_globalpointer --stage serve
```

Модель и tokenizer загружаются из каталога
`exps/exp_xlmr_large_globalpointer/artifacts/model/`. После появления сообщения
о запуске Uvicorn оставьте этот терминал открытым, а проверки выполняйте во
втором терминале.

### 4. Проверка состояния сервиса

Windows PowerShell:

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/healthz"
```

Linux:

```bash
curl --request GET "http://localhost:8000/healthz"
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

### 5. Проверка `predict`

Ручка принимает непустой JSON-массив. Каждый объект должен содержать уникальный
`hash` и исходный `text`:

Windows PowerShell:

```powershell
$RequestBody = @'
[
  {
    "hash": "example-001",
    "text": "Aziza Karimova Google kompaniyasida ishlaydi."
  },
  {
    "hash": "example-002",
    "text": "Ali Toshkentga bordi."
  }
]
'@

$Response = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/api/v1/predict" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($RequestBody))

$Response | ConvertTo-Json -Depth 10
```

Linux:

```bash
curl --request POST "http://localhost:8000/api/v1/predict" \
  --header "Content-Type: application/json; charset=utf-8" \
  --data-binary '[
    {
      "hash": "example-001",
      "text": "Aziza Karimova Google kompaniyasida ishlaydi."
    },
    {
      "hash": "example-002",
      "text": "Ali Toshkentga bordi."
    }
  ]'
```

Пример ответа корректно обученной модели:

```json
{
  "data": [
    {
      "hash": "example-001",
      "entities": [
        {"label": "NAME", "start": 0, "end": 14},
        {"label": "ORG", "start": 15, "end": 21}
      ]
    },
    {
      "hash": "example-002",
      "entities": [
        {"label": "NAME", "start": 0, "end": 3},
        {"label": "GEO", "start": 4, "end": 14}
      ]
    }
  ]
}
```

Конкретный набор найденных сущностей зависит от обученных весов. 

## HTTP API

Доступные endpoints:

- `GET /healthz`;
- `POST /api/v1/predict`;
- `POST /api/v1/predict/normalized`.

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
