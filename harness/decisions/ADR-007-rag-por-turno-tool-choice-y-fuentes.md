# ADR-007 — RAG por turno, tool obligatoria, fuera de alcance y fuente «pedidos»

- Estado: aceptada
- Fecha: 2026-10-03

## Contexto
Una prueba real con Claude Haiku mostró cuatro fallas del orquestador: (1) el modelo no llamaba `buscar_politicas`, así que T10 rechazaba su respuesta correcta (licuadora: 6 meses citando doc1) por `fuente_no_recuperada` y `cifra_sin_sustento`; (2) a veces respondía con texto suelto en vez de usar tools; (3) una pregunta ajena («¿Quién ganó el mundial?») terminaba en «problema técnico»; (4) al usar la tool de pedidos no había una fuente válida que citar.

## Decisión
- **RAG determinista por turno.** El orquestador ejecuta el retriever con el mensaje actual en cada turno e inyecta los fragmentos (ids = `doc_id` real, neutralizados) en un segundo mensaje `system` justo después del prompt. `buscar_politicas` queda como tool auxiliar: sus resultados se suman a la evidencia del turno sin duplicar chunks. T10 verifica contra los chunks de ESE turno (no se arrastran entre turnos). Un fallo del retriever es fallo seguro (escalar).
- **`tool_choice` en el puerto `LLMClient.completar`** (parámetro opcional, retrocompatible, valores abstractos `None`/`"auto"`/`"any"`). Anthropic: `{"type":"any"}`; Azure: `"required"`; un valor desconocido es `ValueError`. Se envía solo si hay `tools`. El bucle del orquestador usa `"any"`; el clasificador no (ya fuerza su propia tool). `FakeLLM` lo registra. Si aun así llega texto suelto, se reintenta una vez con el recordatorio y el tope `max_iteraciones_llm` sigue acotando el bucle.
- **Fuera de alcance.** El contexto inyectado dice explícitamente «No se recuperaron documentos relevantes para este mensaje» y `sistema.md` define qué hacer: respuesta amable que explique en qué sí se puede ayudar, sin escalar ni citar fuentes. Si el LLM da solo texto suelto tras el recordatorio, y además no intentó ninguna tool, no hubo documentos ni pedido en el turno y el mensaje no menciona un id de pedido, el código usa `MENSAJE_FUERA_DE_ALCANCE` con `responder`. Iteraciones agotadas o tool-calls inválidos repetidos nunca se tratan como fuera de alcance (son falla real). «Problema técnico» queda solo para fallas reales (error del LLM, respuesta vacía, retriever, tool, excepción, `responder` duplicado).
- **Fuente «pedidos».** `verificar_salida` acepta «pedidos» solo si en el turno la tool devolvió un pedido real (sin clave `error`); nunca sin llamada ni con error. `AgentResponse.fuentes` ya admite cualquier texto. El prompt y la tool `responder` indican citarla.

## Alternativas consideradas
- Dejar el RAG como decisión del LLM (con prompt más insistente): no determinista, es lo que falló.
- Fundir el contexto en el prompt del sistema: mezcla datos recuperados con instrucciones; un mensaje aparte los mantiene separados (Anthropic igualmente los une en `system`).
- Escalar los temas ajenos: mala experiencia y satura al equipo humano.
- Relajar T10 para aceptar fuentes no recuperadas: abre la puerta a citar documentos inventados.

## Consecuencias
- Positivas: respuestas correctas ya no dependen de que el modelo decida buscar; la verificación sigue siendo estricta por turno; menos fallos «técnicos» falsos.
- Negativas: el retriever corre siempre (costo mínimo con BM25); la consulta es solo el mensaje actual, así que un seguimiento sin contexto («¿y para lavadoras?») puede no recuperar nada y el LLM deberá usar `buscar_politicas`; la plantilla de fuera de alcance también cubre una pregunta legítima sin sustento (sin id de pedido) si el LLM da texto suelto dos veces (dice en qué puede ayudar, sin inventar).
