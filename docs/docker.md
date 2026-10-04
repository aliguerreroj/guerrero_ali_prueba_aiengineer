# Docker y atajos de comandos

Una sola imagen (`python:3.13-slim`, usuario no root `app`, uid 10001) sirve para la CLI, la API
y el servidor MCP. Incluye el modelo de embeddings de fastembed, descargado **durante el build**
en `/opt/fastembed_cache` (variable `FASTEMBED_CACHE_PATH`); en ejecución `HF_HUB_OFFLINE=1`, así
que el contenedor no necesita red para el retriever híbrido.

## Construir

```
docker build -t tiendahogar-agent .
```

Etapas del `Dockerfile`: `builder` (dependencias + modelo), `test` (añade pytest, ruff y el repo
completo) y `runtime` (la final: ligera, sin dependencias de desarrollo ni tests).

## La API key nunca va en la imagen

`.dockerignore` excluye `.env` y `harness/context/`. La clave se pasa al ejecutar:

```
docker run --rm -it --env-file .env tiendahogar-agent
docker run --rm -it -e ANTHROPIC_API_KEY=<clave> -e LLM_PROVIDER=anthropic tiendahogar-agent
```

Sin clave, usa `-e LLM_PROVIDER=fake` (respuestas de demostración, sin red).

## CLI interactiva (CMD por defecto)

```
docker run --rm -it -e LLM_PROVIDER=fake tiendahogar-agent
echo "¿Cuánto dura la garantía?" | docker run -i --rm -e LLM_PROVIDER=fake tiendahogar-agent
```

## API (puerto 8080 dentro y fuera)

```
docker run --rm -p 8080:8080 -e LLM_PROVIDER=fake tiendahogar-agent \
  python -m uvicorn tiendahogar_agent.api:app --host 0.0.0.0 --port 8080
```

Prueba: `curl http://localhost:8080/health`. Cuerpo de `/chat`: `{"conversation_id": "c1", "mensaje": "¿Cuánto dura la garantía?"}` (campos `conversation_id` y `mensaje`). Detalles de la API en `docs/api.md`.

## Servidor MCP (stdio)

```
docker run -i --rm tiendahogar-agent python -m tiendahogar_agent.mcp_server
```

## Tests dentro del contenedor

```
docker build --target test -t tiendahogar-agent:test .
docker run --rm -e LLM_PROVIDER=fake tiendahogar-agent:test
```

La etapa `test` usa los datos de `data/docs/` y `harness/docs_checksums.json` copiados del contexto
(`.gitattributes` normaliza a LF y el checksum normaliza CRLF a LF). Un único test, el que usa
`git check-ignore`, se omite porque la imagen no tiene git ni el directorio `.git`.

## Makefile y equivalentes sin make

En Windows no siempre hay `make`. Cada target tiene su equivalente. En PowerShell activa antes el
entorno (`.venv\Scripts\Activate.ps1`); en bash, `source .venv/bin/activate` (o
`.venv/Scripts/activate` en Git Bash). Con el Makefile, la variable `PYTHON` detecta el `.venv`
solo y se puede forzar: `make test PYTHON=/ruta/python`.

| Target | PowerShell | bash |
|---|---|---|
| `install` | `python -m pip install -e ".[dev]"` | `python -m pip install -e ".[dev]"` |
| `test` | `python -m pytest tests/ -q` | `python -m pytest tests/ -q` |
| `lint` | `python -m ruff check .` | `python -m ruff check .` |
| `eval` (llamadas REALES, cuesta dinero; requiere `ANTHROPIC_API_KEY`) | `python -m tiendahogar_agent.evals` | `python -m tiendahogar_agent.evals` |
| `chat` | `python -m tiendahogar_agent.cli` | `python -m tiendahogar_agent.cli` |
| `api` (puerto 8080, `PUERTO=` para cambiarlo) | `python -m uvicorn tiendahogar_agent.api:app --port 8080` | `python -m uvicorn tiendahogar_agent.api:app --port 8080` |
| `docker-build` | `docker build -t tiendahogar-agent .` | `docker build -t tiendahogar-agent .` |
| `docker-run` | `docker run --rm -it --env-file .env tiendahogar-agent` | `docker run --rm -it --env-file .env tiendahogar-agent` |

Entre comillas siempre las rutas: el repositorio puede estar en una carpeta con espacios.
