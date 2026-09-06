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

# Accept the flat token-classification bundle or the nested GlobalPointer
# bundle. Both checks guarantee that startup never needs a model download.
RUN model_dir="/app/exps/${EXP_NAME}/artifacts/model"; \
    flat_bundle=false; \
    pointer_bundle=false; \
    if [ -f "${model_dir}/config.json" ] && \
       [ -f "${model_dir}/tokenizer_config.json" ]; then \
      flat_bundle=true; \
    fi; \
    if [ -f "${model_dir}/model_config.json" ] && \
       [ -f "${model_dir}/pointer_state.pt" ] && \
       [ -f "${model_dir}/encoder/config.json" ] && \
       [ -f "${model_dir}/tokenizer/tokenizer_config.json" ] && \
       { [ -f "${model_dir}/encoder/model.safetensors" ] || \
         [ -f "${model_dir}/encoder/pytorch_model.bin" ]; }; then \
      pointer_bundle=true; \
    fi; \
    if [ "${flat_bundle}" != true ] && [ "${pointer_bundle}" != true ]; then \
      echo "ERROR: inference model bundle is missing for ${EXP_NAME}" >&2; \
      exit 1; \
    fi

EXPOSE 8000

ENTRYPOINT ["uv", "run", "python", "main.py"]
CMD ["--stage", "serve"]
