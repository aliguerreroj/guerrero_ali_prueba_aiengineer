# Imagen del agente de soporte de TiendaHogar.
#
# Etapas:
#   builder  -> instala dependencias en un venv y descarga el modelo de embeddings (con red).
#   test     -> añade pytest/ruff y el repo completo; solo se usa con --target test.
#   runtime  -> etapa final (la que construye `docker build .`): ligera, sin dev, usuario no root.
#
# La API key NUNCA se copia a la imagen (.dockerignore excluye .env): se pasa al ejecutar con
# -e ANTHROPIC_API_KEY=<clave> o --env-file .env.

# ---------------------------------------------------------------- builder
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalación editable: RAIZ_REPO del código se calcula con parents[2] = /app.
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# Descarga del modelo de embeddings DURANTE EL BUILD (el modelo sale de settings.yaml).
COPY settings.yaml ./
RUN python -c "from tiendahogar_agent.config import cargar_settings; \
from fastembed import TextEmbedding; \
m = cargar_settings().embedding_model; \
v = list(TextEmbedding(model_name=m).embed(['prueba'])); \
print('modelo descargado:', m, len(v[0]), 'dimensiones')"

# ---------------------------------------------------------------- test
FROM python:3.13-slim AS test

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    LLM_PROVIDER=fake \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/fastembed_cache /opt/fastembed_cache
WORKDIR /app
# Repo completo (el .dockerignore deja fuera .env, harness/context, .git, cachés, logs).
COPY --chown=app:app . .
# harness/init.py busca el entorno en .venv: se enlaza al venv de la imagen.
RUN pip install pytest ruff "httpx>=0.27" && ln -s /opt/venv /app/.venv && mkdir -p logs && chown app:app logs /app
USER app
CMD ["python", "-m", "pytest", "tests/", "-q"]

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    HF_HUB_OFFLINE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
# El modelo ya está en la imagen: en ejecución no se descarga nada (HF_HUB_OFFLINE=1).
COPY --from=builder --chown=app:app /opt/fastembed_cache /opt/fastembed_cache

WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
COPY --chown=app:app evals ./evals
COPY --chown=app:app settings.yaml pyproject.toml ./
COPY --chown=app:app harness/docs_checksums.json ./harness/docs_checksums.json
RUN mkdir -p logs && chown app:app logs /app

USER app
EXPOSE 8080
# Por defecto, la CLI de chat (usa -it para interactuar). Otras entradas, con la misma imagen:
#   API:  python -m uvicorn tiendahogar_agent.api:app --host 0.0.0.0 --port 8080
#   MCP:  python -m tiendahogar_agent.mcp_server
CMD ["python", "-m", "tiendahogar_agent.cli"]
