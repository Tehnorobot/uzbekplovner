FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

ARG DEFAULT_EXP=exp_baseline
ENV EXP_NAME=${DEFAULT_EXP}

COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv==0.8.5 && uv sync --extra dev --no-install-project

COPY main.py .
COPY src ./src
COPY scripts ./scripts
COPY exps ./exps

# The current experiment model is intentionally checked during build. This
# prevents an image that downloads a model after startup.
RUN if [ ! -d "/app/exps/${EXP_NAME}/artifacts/model" ] || \
       [ ! -f "/app/exps/${EXP_NAME}/artifacts/model/config.json" ] || \
       [ ! -f "/app/exps/${EXP_NAME}/artifacts/model/tokenizer_config.json" ]; then \
      echo "ERROR: model is missing for ${EXP_NAME}; run training first" >&2; \
      exit 1; \
    fi

EXPOSE 8000

ENTRYPOINT ["uv", "run", "python", "main.py"]
CMD ["--stage", "serve"]
