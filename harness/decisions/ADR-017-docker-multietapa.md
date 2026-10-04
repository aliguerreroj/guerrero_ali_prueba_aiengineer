# ADR-017 — Docker multietapa y atajos con Makefile

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
Quien evalúe debe poder ejecutar el agente sin preparar un entorno Python, y la imagen no debe necesitar red en ejecución (el modelo de embeddings, ADR-013, se descarga una vez). La API key no puede acabar dentro de una imagen. Se documenta en `docs/docker.md`.

## Decisión
`Dockerfile` con tres etapas sobre `python:3.13-slim`:
- **builder**: crea el venv en `/opt/venv`, instala el paquete con `pip install -e .` y descarga el modelo de embeddings **durante el build** (el nombre sale de `settings.yaml`) en `/opt/fastembed_cache` (`FASTEMBED_CACHE_PATH`), imprimiendo sus dimensiones.
- **test**: añade `pytest`, `ruff` y `httpx`, el repo completo y `LLM_PROVIDER=fake`; se usa con `--target test` y ejecuta `pytest tests/ -q`.
- **runtime** (la etapa final de `docker build .`): copia solo el venv, el modelo, `src`, `data`, `evals`, `settings.yaml`, `pyproject.toml` y `harness/docs_checksums.json`. Sin dependencias de desarrollo ni tests. `HF_HUB_OFFLINE=1`: en ejecución no se descarga nada.
- **Usuario no root**: `app` con uid 10001 en `test` y `runtime`.
- **`.env` nunca en la imagen**: `.dockerignore` excluye `.env`, `.env.*` (salvo `.env.example`) y `harness/context/`, además de `.git/`, cachés, `logs/` y `evals/resultados/`. La clave se pasa al ejecutar con `--env-file .env` o `-e ANTHROPIC_API_KEY=...`.
- **Una imagen para tres usos**: CMD por defecto = CLI (`python -m tiendahogar_agent.cli`); API con `python -m uvicorn tiendahogar_agent.api:app --host 0.0.0.0 --port 8080` (`EXPOSE 8080`); MCP con `python -m tiendahogar_agent.mcp_server`.
- **Makefile**: atajos `install`, `test`, `lint`, `eval`, `chat`, `api`, `docker-build` y `docker-run`, con la variable `PYTHON` que detecta el `.venv` y rutas entre comillas (el repo puede estar en una carpeta con espacios). `docs/docker.md` da el equivalente en PowerShell y bash para quien no tenga `make`. `eval` hace llamadas reales y cuesta dinero.

## Alternativas consideradas
- Una sola etapa: la imagen final cargaría herramientas de build y de desarrollo.
- Descargar el modelo al primer arranque: el contenedor necesitaría red y el primer arranque sería lento y frágil.
- Una imagen por servicio (CLI, API, MCP): triplica el mantenimiento sin diferencias de dependencias.
- Ejecutar como root: más simple, pero innecesario y menos seguro.

## Consecuencias
- Positivas: arranque sin red; imagen final sin dependencias de desarrollo; sin secretos en las capas; pruebas reproducibles dentro del contenedor.
- Negativas: el build exige red y descarga el modelo (~0,22 GB, ADR-013) y la imagen es grande por ello; en la etapa `test` se omite un único test que usa `git` (la imagen no tiene `.git`); el historial de la API es en memoria (ADR-015), por lo que escalar a varias réplicas necesitaría un almacén externo.
