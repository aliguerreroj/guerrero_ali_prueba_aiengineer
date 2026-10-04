# TiendaHogar — Agente de soporte

[![CI](https://github.com/aliguerreroj/guerrero_ali_prueba_aiengineer/actions/workflows/ci.yml/badge.svg)](https://github.com/aliguerreroj/guerrero_ali_prueba_aiengineer/actions/workflows/ci.yml)

Agente de soporte al cliente de «TiendaHogar», una empresa ficticia de electrodomésticos. Responde dudas de garantía, devoluciones, envíos, reembolsos y canales con RAG (recuperación híbrida BM25 + embeddings) sobre 5 documentos, consulta el estado de un pedido con la tool `consultar_estado_pedido(order_id: str) -> dict` sobre una tabla mock y, mediante guardrails en capas, escala a soporte@tiendahogar.example lo que los documentos prohíben manejar (reembolsos por encima del umbral, quejas sobre el trato de un empleado, disputas de facturación y temas legales). Si no hay sustento en los documentos o en la tabla de pedidos, lo dice o escala; nunca inventa.

## Requisitos

- Python 3.11 o superior. El CI prueba Python 3.11 y 3.13 en Ubuntu y Windows.
- Con `LLM_PROVIDER=fake` (por defecto) no necesitas red ni API key: la CLI usa solo BM25 y no descarga ningún modelo.
- **Aviso de descarga:** sin Docker, la primera ejecución que use embeddings descarga un modelo de unos 0,22 GB. Eso ocurre con `LLM_PROVIDER=anthropic` o `azure`, con la calibración y con los evals. En Docker el modelo se descarga durante el build.
- Los tests por defecto no descargan nada: `pyproject.toml` fija `-m 'not integration'`, y solo 2 tests marcados `integration` (modelo de embeddings real y llamada real mínima al LLM) usan red.

## Instalación

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Si PowerShell bloquea la activación por la política de ejecución, habilita scripts solo para la sesión actual con `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` y vuelve a activar.

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Las rutas del repo pueden tener espacios: entrecomíllalas siempre.

## Variables de entorno

`.env` está ignorado por git; nunca subas claves. Las variables se leen desde el entorno o desde `.env` (el entorno gana). El resto de ajustes viven en `settings.yaml`.

| Variable | Valores / por defecto | Cuándo es obligatoria |
|---|---|---|
| `LLM_PROVIDER` | `fake` (por defecto), `anthropic`, `azure` | Nunca; elige el modo |
| `ANTHROPIC_API_KEY` | vacía | Con `LLM_PROVIDER=anthropic` y para los evals |
| `AZURE_OPENAI_API_KEY` | vacía | Solo con `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_ENDPOINT` | vacía | Solo con `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_DEPLOYMENT` | vacía (nombre del deployment) | Solo con `LLM_PROVIDER=azure` |
| `USAR_CLASIFICADOR_LLM` | `true` (la CLI lo apaga sola con `fake`) | Opcional |
| `MAX_ITERACIONES_LLM` | `5` (entero >= 1; llamadas al LLM por turno) | Opcional |
| `TIENDAHOGAR_LOGS_DIR` | `logs/` en la raíz del repo | Opcional |

Con `fake` no necesitas ninguna variable.

## Chatear con el agente (CLI)

Modo `fake` (sin red ni clave; respuestas sencillas pero reales, citan los documentos o la tabla de pedidos):

```
python -m tiendahogar_agent.cli
python -m tiendahogar_agent.cli --una-vez "¿Cuánto dura la garantía de una lavadora?"
tiendahogar-chat
```

Se sale con «salir», «exit», Ctrl+D o Ctrl+C. Tras cada respuesta se muestra la acción, las fuentes, el canal (si escala) y el `trace_id`. Preguntas de ejemplo:

- «¿Cuánto dura la garantía de una lavadora?» → responde con la política de garantía (doc1).
- «¿Dónde está mi pedido ORD-1001?» → consulta la tabla de pedidos.
- «Quiero un reembolso de 800» → escala a soporte@tiendahogar.example.

Con LLM real (Anthropic), ponlo en el `.env` (`LLM_PROVIDER=anthropic` y `ANTHROPIC_API_KEY=...`) o solo en la sesión:

```powershell
$env:LLM_PROVIDER="anthropic"
python -m tiendahogar_agent.cli
```

```bash
export LLM_PROVIDER=anthropic
python -m tiendahogar_agent.cli
```

## API HTTP

```
python -m uvicorn tiendahogar_agent.api:app --port 8000
```

Ejemplo con curl (bash):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "mensaje": "¿Dónde está mi pedido ORD-1001?"}'
```

Ejemplo con PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -ContentType "application/json; charset=utf-8" -Body (@{conversation_id="demo-1"; mensaje="¿Cuánto dura la garantía?"} | ConvertTo-Json)
```

También hay `GET /health` y Swagger en <http://localhost:8000/docs>. Detalle en [`docs/api.md`](docs/api.md).

## Servidor MCP

Expone la misma tool `consultar_estado_pedido` por stdio:

```
python -m tiendahogar_agent.mcp_server
```

Detalle en [`docs/mcp.md`](docs/mcp.md).

## Pruebas

```
python -m pytest tests/
python -m ruff check .
python harness/init.py
```

`python -m pytest tests/ --collect-only -q` informa «1506/1508 tests collected (2 deselected)»: 1506 tests por defecto, deterministas (FakeLLM, sin red ni API key), y 2 de integración deshabilitados por defecto. Esos 2 se ejecutan a propósito con `python -m pytest -m integration`; usan red (descargan el modelo de embeddings y hacen una llamada mínima real al LLM, que requiere clave).

`python harness/init.py` verifica el entorno, los checksums de los documentos, ruff y pytest.

## Evals y calibración

```
python -m tiendahogar_agent.evals
python -m tiendahogar_agent.calibracion
```

- **Evals:** ejecutan el golden set con el LLM real. **Hacen llamadas reales y cuestan dinero.** Requieren `ANTHROPIC_API_KEY`; sin clave se omiten con un aviso. El tope de costo es `--max-costo` (2 USD por defecto).
- **Calibración:** mide los puntajes de relevancia del retriever para ajustar los umbrales. No llama al LLM; usa el embedder local (descarga el modelo la primera vez).

Último reporte real (`evals/resultados/2026-10-04-1140-claude-haiku-4-5-20251001.md`, modelo `claude-haiku-4-5-20251001`): 37 casos, 76 % de caso OK, acción correcta en el 92 %, costo 0.1718 USD. Es una sola ejecución y el costo es una estimación con precios tomados de `settings.yaml`, no verificados en vivo. Más en [`evals/README.md`](evals/README.md).

## Docker y Makefile

Resumen (detalle y equivalentes de cada target para PowerShell y bash en [`docs/docker.md`](docs/docker.md)):

```
docker build -t tiendahogar-agent .
docker run -it --rm --env-file .env tiendahogar-agent                      # CLI de chat
docker run --rm -p 8080:8080 --env-file .env tiendahogar-agent python -m uvicorn tiendahogar_agent.api:app --host 0.0.0.0 --port 8080
```

Targets de `make`: `install`, `test`, `lint`, `eval`, `chat`, `api`, `docker-build` y `docker-run`. La API key nunca se copia a la imagen: se pasa con `--env-file` o `-e`.

## Estructura del repo

```
src/tiendahogar_agent/   código de la aplicación (dominio, puertos, adaptadores, CLI, API, MCP)
tests/                   pruebas pytest
data/docs/               los 5 documentos base (intocables)
evals/                   golden set, preguntas fuera de dominio y resultados
docs/                    api, mcp, docker y guía de tono
harness/                 init.py, tasks.json, progress/, decisions/ (ADRs), checksums
.claude/                 subagentes, hooks y settings de Claude Code
.github/workflows/       CI (ci.yml)
Dockerfile, Makefile
settings.yaml            ajustes no secretos
pyproject.toml
.env.example
```

## Documentación

- [`docs/api.md`](docs/api.md): API HTTP.
- [`docs/mcp.md`](docs/mcp.md): servidor MCP.
- [`docs/docker.md`](docs/docker.md): Docker y Makefile.
- [`docs/guia_de_tono.md`](docs/guia_de_tono.md): guía de tono del agente.
- [`harness/decisions/README.md`](harness/decisions/README.md): índice de decisiones de arquitectura (ADRs).
- [`evals/README.md`](evals/README.md): evals y calibración.
- [`SUBMISSION.md`](SUBMISSION.md): resumen de la entrega.

## Cómo se construyó

El proyecto se desarrolló con asistencia de IA (Claude Code) bajo supervisión humana. El humano define los criterios y valida; los agentes implementan. Para que el proceso sea verificable:

- [`AGENTS.md`](AGENTS.md) fija las reglas del repo y el flujo de trabajo.
- `harness/tasks.json` lista las tareas con sus criterios y su verificación; los agentes solo actualizan estado y notas.
- `harness/progress/` guarda un registro por sesión.
- Hooks en `.claude/hooks/`: `protect_files` bloquea editar los documentos, sus checksums, el contexto privado y `.env`; `pre_commit_gate` ejecuta `harness/init.py` antes de cada `git commit` y lo bloquea si falla; `session_context` muestra al iniciar la sesión los últimos commits, el último registro de progreso y las tareas pendientes.
- `harness/init.py` comprueba el entorno, los checksums de los documentos, `ruff` y `pytest`.
- Cada tarea sigue el flujo lector → implementador → revisor, con subagentes en `.claude/agents/`; el revisor trabaja con contexto limpio y puede rechazar.
