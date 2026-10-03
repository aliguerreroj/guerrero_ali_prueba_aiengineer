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

## Nota de la ronda 2: reconsulta determinista por turno y retiro de cita sin respaldo (2026-10-03)
Dos regresiones de la prueba en vivo (ambas `fuente_no_recuperada`): (1) «¿Y el pedido ORD-9999?»: el LLM explicaba bien que no existe pero citaba «pedidos» como pide el prompt, y T10 solo acepta esa fuente con un pedido real; (2) «pero lo necesito urgente» tras consultar ORD-1003: la evidencia es por turno, así que en el segundo turno no había pedido que sustentara «pedidos» ni los 6 días hábiles.
- **Reconsulta por turno.** Antes del LLM el orquestador extrae los ids `ORD-####` (regex estricto, normalizados con el helper de `pedidos.py`) del mensaje y de los mensajes `user` y `assistant` del historial (máx. 3, los más recientes) y los consulta en el repositorio. Solo se toma el id del historial, nunca lo que dijo el asistente: el repositorio es la fuente de verdad. Los resultados entran en la evidencia del turno y en un mensaje `system` de contexto («consulta de pedido ya realizada por el sistema»). Un fallo del repositorio es fallo seguro (escalar).
- **Retiro de cita sin respaldo.** Al entregar, si el LLM cita «pedidos» y hubo consultas pero todas con error, la cita se retira antes de verificar (quitar una cita sin respaldo no inventa nada). Si nunca hubo consulta, la cita sigue siendo `fuente_no_recuperada`. `verificar_salida` no cambia.
- **pedir_dato forzado.** Si el turno (mensaje actual o tools del LLM) solo tuvo ids inexistentes o inválidos y no se cita ninguna fuente, la acción sube a `pedir_dato` aunque el LLM no lo sugiera: pedir revisar el número es lo único coherente y está entre `responder` y `escalar`. No se fuerza si el error viene solo del historial ni si se citan documentos.
- Consecuencia: un id escrito en el mensaje se consulta siempre, aunque el LLM no use la tool; coste mínimo (tabla en memoria).
