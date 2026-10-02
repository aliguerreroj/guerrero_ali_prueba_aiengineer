# AGENTS.md — instrucciones para cualquier LLM que trabaje en este repo

Todo (código, comentarios, commits, documentación) va **en español**.

## Misión del proyecto
Construir un agente de soporte al cliente para «TiendaHogar» (empresa ficticia de electrodomésticos). El agente:
- responde dudas de garantía, devoluciones, envíos, reembolsos y canales usando RAG sobre los 5 documentos de `data/docs/`;
- consulta el estado de un pedido con la tool `consultar_estado_pedido` sobre una tabla mock;
- escala a un humano (soporte@tiendahogar.example) los casos que los documentos prohíben manejar: reembolsos por encima del umbral (500), quejas sobre el trato de un empleado, disputas de facturación y temas legales;
- nunca inventa información que no esté en los documentos o en la tabla de pedidos;
- se valida con tests automatizados (pytest) deterministas.

Plazo de entrega: **2026-10-05 12:00 (hora Colombia, UTC-5)**. Se evalúa el último commit dentro del plazo.

## Reglas no negociables
1. La firma de la tool es exactamente `consultar_estado_pedido(order_id: str) -> dict`.
2. **Prohibido editar `data/docs/`** (ni `harness/docs_checksums.json`). Los hooks lo bloquean.
3. El agente nunca inventa información: si no hay sustento, lo dice o escala.
4. Nada de secretos en el repo: usar `.env` (ignorado por git) y mantener `.env.example` actualizado.
5. `harness/context/` (material privado del enunciado) no se publica ni se copia literalmente a archivos versionados; descríbelo con tus propias palabras.
6. Todo en español.

## Estructura del repo
| Ruta | Propósito |
|---|---|
| `src/tiendahogar_agent/` | Código de la aplicación (obligatorio por el enunciado) |
| `tests/` | Pruebas pytest (obligatorio) |
| `README.md`, `SUBMISSION.md` | Entregables obligatorios en la raíz |
| `data/docs/` | Los 5 documentos base, intocables |
| `harness/init.py` | Verificación del entorno y del repo |
| `harness/tasks.json` | Lista de tareas y su estado |
| `harness/progress/` | Un registro por sesión |
| `harness/decisions/` | ADRs (decisiones de arquitectura) |
| `harness/docs_checksums.json` | SHA-256 de los documentos (CRLF normalizado a LF) |
| `harness/context/` | Enunciado privado, ignorado por git |
| `.claude/` | Subagentes, hooks y settings de Claude Code |
| `logs/` | Trazas en runtime, ignoradas por git |

## Comandos
Crear y activar el entorno (una vez):
```
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate
python -m pip install -e ".[dev]"
```
Verificar y probar (con el .venv activado):
```
python harness/init.py          # verificación completa
python harness/init.py --full   # reservado para evals (aún no implementados)
python -m pytest tests/ -q
python -m ruff check .
```
Las rutas del repo pueden tener espacios: entrecomíllalas siempre.

## Ritual de inicio de sesión
1. Leer este `AGENTS.md`.
2. `git log --oneline -10`.
3. Leer el último archivo de `harness/progress/` (excluyendo `README.md`).
4. Revisar tareas pendientes en `harness/tasks.json`.
5. Correr `python harness/init.py`.
6. Solo entonces elegir **UNA** tarea (respetando `depende_de`).

## Flujo multiagente por tarea
1. **Principal** elige la tarea.
2. **Lector** (`.claude/agents/lector.md`) resume el contexto relevante.
3. **Implementador** escribe los tests primero, luego el código mínimo, y corre `python harness/init.py`.
4. **Revisor** (contexto limpio, adversarial) responde APROBADO o RECHAZADO con hallazgos.
5. Si rechaza, vuelve al Implementador. Si aprueba, el Principal actualiza `tasks.json`, escribe el registro en `harness/progress/` y hace commit.

## Definición de «terminado»
La verificación de la tarea pasa, `python harness/init.py` está en verde y el revisor aprobó.

## Reglas de `tasks.json`
- Los agentes **solo** pueden modificar `estado`, `revisado_por_revisor` y `notas`.
- Nunca borrar tareas ni reescribir `criterios`, `verificacion`, `prioridad` o `depende_de`: eso lo decide el humano.
- Estados: `pendiente` → `en_progreso` → `hecha`.

## Principios de diseño de la app
- **Puertos y adaptadores**: el dominio depende de interfaces (LLMClient, Embedder, VectorStore, DocumentSource, OrderRepository, EventBus); los SDKs viven en adaptadores.
- **Guardrails en capas**: reglas deterministas → clasificador LLM → verificación de salida.
- **El sistema decide, el LLM redacta**: escalar/rechazar lo decide código, no el modelo.
- **Tono**: natural, empático, orientado a solución, tuteo.
- **Tests deterministas** con `FakeLLM`: sin API key ni red.
- Fallo seguro = escalar a humano.

## Convención de commits
Prefijos: `feat:`, `fix:`, `test:`, `docs:`, `chore:`. Mensaje en español, en imperativo o descriptivo y breve. Un commit por tarea terminada. El hook `pre_commit_gate` ejecuta `harness/init.py` y bloquea el commit si falla. No hacer `push` sin que el humano lo pida.
