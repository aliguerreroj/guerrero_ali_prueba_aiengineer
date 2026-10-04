# ADR-012 — Guardrails en capas: el sistema decide, el LLM redacta

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
Los documentos prohíben al agente manejar ciertos casos (reembolsos por encima del umbral, quejas sobre el trato de un empleado, disputas de facturación, temas legales), y el agente nunca debe inventar datos ni asumir compromisos. Un modelo de lenguaje puede equivocarse, así que la decisión de qué hacer no puede depender solo de su criterio. Este ADR describe el orden real de las capas en `Orquestador.procesar` y qué decide cada una; el detalle de cada regla está en los ADR enlazados y no se repite aquí.

## Decisión
**La acción (`responder` | `escalar` | `pedir_dato`) la decide el código; el LLM solo redacta el texto.** Orden real de las capas por turno (`orquestador.py`):
1. **Entrada no válida**: mensaje no textual o vacío visible: `pedir_dato` con plantilla, sin LLM.
2. **Reglas deterministas de entrada** (`guardrail_input.py`, sin LLM): umbral de reembolso (`umbral_reembolso`, 500 en `settings.yaml`), queja de trato, facturación, legal y manipulación; una sola categoría por precedencia.
3. **Clasificador LLM** (`clasificador.py`; se omite si las reglas ya escalaron y se desactiva con `usar_clasificador_llm`): salida estructurada por una tool con enum estricto. **Solo puede añadir escalamientos**; nunca anula una decisión de las reglas y cualquier fallo suyo devuelve «sin clasificación» y se sigue con las reglas (ADR-008). Si clasifica `fuera_de_alcance`, el orquestador responde con una plantilla determinista, sin bucle ni fuentes.
4. **Escalamiento**: si algo escaló, el LLM redacta el mensaje sin tools, `verificar_salida` lo revisa y, si el LLM falla o el texto no cumple, se usa una plantilla de respaldo por categoría.
5. **Contexto recuperado y bucle de tool use**: el retriever corre en cada turno (ADR-007) y el LLM responde mediante tools con `tool_choice="any"`; la acción base es `responder` y la `accion_sugerida` del LLM solo se acepta si va hacia el lado más seguro (orden `responder` < `pedir_dato` < `escalar`, función `accion_mas_segura`). Citar `doc5` sube la acción a `escalar` salvo que el cliente pregunte por los canales (ADR-008).
6. **Verificación de salida** (`guardrail_output.verificar_salida`, determinista, siempre antes de entregar): cifras con sustento, fuentes realmente recuperadas, canal de escalamiento cuando la acción es `escalar`, longitud máxima, `canal_innecesario`, compromisos (`guardrail_compromisos.py`, ADR-005) y hechos críticos (`hechos.py`, ADR-009). Si falla, se entrega una respuesta segura con acción `escalar`. Para `hecho_incorrecto` y `canal_innecesario` el orquestador hace **un solo reintento** pidiendo la corrección antes de caer a la respuesta segura.
7. **Fallo seguro** (`resiliencia.py`, ADR-006), transversal: error del LLM tras los reintentos del SDK, respuesta vacía, error interno de la tool de pedidos, fallo del retriever, bucle sin respuesta o cualquier excepción inesperada en el borde de `procesar` producen `escalar` con el canal de soporte y un texto sin cifras ni compromisos.

Cada escalamiento, sea cual sea la capa que lo originó, publica el evento de ADR-010 al final del turno.

## Alternativas consideradas
- Dejar que el LLM decida la acción con un prompt estricto: no es determinista ni auditable, y una inyección de prompt podría evitar un escalamiento obligatorio.
- Solo reglas, sin clasificador: pierde las paráfrasis que las reglas léxicas no cubren; el clasificador añade cobertura sin poder relajar nada.
- Solo clasificador LLM: más flexible, pero sin garantías en los casos que los documentos prohíben manejar.
- Verificar la salida con otro LLM: costo, latencia y no determinismo; la verificación léxica es barata y se prueba con tests.

## Consecuencias
- Positivas: los casos prohibidos escalan aunque el LLM falle; el LLM no puede bajar una acción de seguridad; cada capa se prueba de forma determinista con `FakeLLM`; ante la duda se escala.
- Negativas: las capas léxicas (entrada, compromisos, hechos) son heurísticas con límites documentados en sus módulos; escalar de más es el lado seguro pero degrada la experiencia en algunos casos. ADR-005, ADR-006, ADR-008 y ADR-009 detallan cada pieza y sus límites.
