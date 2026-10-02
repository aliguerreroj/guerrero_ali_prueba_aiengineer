# ADR-001 — Arnés de desarrollo y flujo multiagente

- Estado: aceptada
- Fecha: 2026-10-02

## Contexto
El proyecto se construye con ayuda de LLMs en varias sesiones cortas. Sin memoria persistente ni controles, es fácil perder contexto, editar por error los documentos base o declarar «terminado» algo sin verificar.

## Decisión
Se crea un arnés alrededor del código: `AGENTS.md` como única fuente de instrucciones; `tasks.json` como lista de tareas con criterios verificables; `progress/` como memoria entre sesiones; `init.py` como verificación única; hooks que protegen archivos, exigen init en verde antes de commitear e inyectan contexto al iniciar. El trabajo por tarea sigue Principal → Lector → Implementador (tests primero) → Revisor adversarial → commit.

## Alternativas consideradas
- Un solo agente sin revisión: más rápido, pero sin control independiente de calidad.
- Instrucciones repartidas en varios archivos: se desincronizan.

## Consecuencias
- Más disciplina y trazabilidad; algo de sobrecarga por tarea.
- Los agentes solo modifican `estado`, `revisado_por_revisor` y `notas` en `tasks.json`; el humano decide el resto.
