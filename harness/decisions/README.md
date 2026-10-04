# Registro de decisiones de arquitectura (ADR)

Cada ADR sigue la plantilla `TEMPLATE.md` (Contexto, Decisión, Alternativas consideradas, Consecuencias).

| ADR | Título | Tema |
|---|---|---|
| [001](ADR-001-arnes-multiagente.md) | Arnés de desarrollo y flujo multiagente | Proceso de trabajo |
| [002](ADR-002-init-en-python.md) | `init` en Python en lugar de bash | Proceso de trabajo |
| [003](ADR-003-recuperacion-hibrida.md) | Recuperación híbrida BM25 + fastembed | Recuperación (RAG) |
| [004](ADR-004-regla-de-relevancia.md) | Regla de relevancia del retriever | Recuperación (RAG) |
| [005](ADR-005-deteccion-de-compromisos.md) | Detección de compromisos como red de seguridad | Guardrails |
| [006](ADR-006-resiliencia-y-fallo-seguro.md) | Resiliencia: reintentos del SDK, excepciones específicas y fallo seguro | Guardrails y resiliencia |
| [007](ADR-007-rag-por-turno-tool-choice-y-fuentes.md) | RAG por turno, tool obligatoria, fuera de alcance y fuente «pedidos» | Orquestación |
| [008](ADR-008-accion-coherente-y-clasificador-por-defecto.md) | Acción coherente con el escalamiento y clasificador LLM activado por defecto | Guardrails |
| [009](ADR-009-verificacion-de-hechos-criticos.md) | Verificación de hechos críticos (garantía por categoría y envío por destino) | Guardrails |
| [010](ADR-010-evento-de-escalamiento-idempotente.md) | Evento de escalamiento con clave de idempotencia | Eventos |
| [011](ADR-011-puertos-y-adaptadores.md) | Puertos y adaptadores, dobles de prueba y selección de proveedor | Arquitectura |
| [012](ADR-012-guardrails-en-capas.md) | Guardrails en capas: el sistema decide, el LLM redacta | Guardrails |
| [013](ADR-013-embedder-fastembed-multilingue.md) | Embedder: fastembed con MiniLM multilingüe | Recuperación (RAG) |
| [014](ADR-014-chunking-estrategia-auto.md) | Chunking con estrategia `auto` | Recuperación (RAG) |
| [015](ADR-015-api-http-fastapi.md) | API HTTP con FastAPI | Interfaces |
| [016](ADR-016-servidor-mcp.md) | Servidor MCP por stdio | Interfaces |
| [017](ADR-017-docker-multietapa.md) | Docker multietapa y atajos con Makefile | Despliegue |

## Nota sobre contenido que no se duplica
- El umbral y la regla de relevancia del retriever están solo en ADR-004.
- Los hechos críticos (garantía por categoría, envío por destino) están solo en ADR-009.
Los demás ADR los referencian en lugar de repetirlos.
