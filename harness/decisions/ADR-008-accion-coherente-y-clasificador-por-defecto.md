# ADR-008 — Acción coherente con el escalamiento y clasificador LLM activado por defecto

- Estado: aceptada
- Fecha: 2026-10-03

## Contexto
En una prueba en vivo, «Quiero reportar el trato de un empleado» y «Me atendió pésimo el vendedor y quiero quejarme» recibieron un texto que remitía a soporte, pero con `accion=responder` y `canal=None`. Había tres huecos: (a) las reglas de queja de trato no cubrían esas redacciones, (b) el clasificador LLM (T15) estaba apagado por defecto, (c) el orquestador aceptaba como acción final la que sugería el LLM aunque el texto dijera «escríbele a soporte».

## Decisión
**(a) Reglas de trato (T08).** Nuevas reglas por ventana acotada a una oración (sin backtracking): «reportar/reporte/denunciar» + aspecto del trato (trato, actitud, amabilidad, conducta, forma de hablar, «cómo me habló/trató/atendió») + rol; «me atendió/atendieron/habló mal|pésimo|horrible|fatal» + rol (en cualquier orden, con o sin «quiero quejarme»); «queja» + aspecto ampliado + rol; «hablar con un supervisor» + aspecto + un rol distinto del supervisor. Los falsos positivos tolerados por fallo seguro están en el docstring de `guardrail_input.py` (p. ej. «tengo una queja de cómo me atendió el vendedor» sin juicio explícito, o preguntar cómo reportar el trato de un empleado). Las faltas de ortografía («keja») no se detectan: eso queda al clasificador.

**(b) Clasificador activado por defecto.** `usar_clasificador_llm` pasa a `true` en `config.py`, `settings.yaml` y `.env.example`. Costo: **una llamada LLM extra por cada mensaje que las reglas no escalan** (latencia y tokens; acotada por `max_tokens_clasificador`=96); si el clasificador falla, se sigue con las reglas (no escala por error). Con `LLM_PROVIDER=fake`, la fábrica del CLI (`construir_orquestador`) lo apaga: `LLMDemo` no tiene la tool de clasificación, así que el clasificador solo generaría fallos en los logs. El `Orquestador` en sí no cambia: los tests que inyectan un `FakeLLM` guionado fijan `usar_clasificador_llm` explícitamente (los que no lo usan lo ponen en `False` para que el guion solo cubra el bucle).

**(c) Acción coherente con el texto (`Orquestador._entregar`).** Si la respuesta del LLM remite al canal humano —el texto contiene `soporte@tiendahogar.example` (sin tildes, sin distinguir mayúsculas) o cita `doc5`—, la acción pasa a `escalar` con `accion_mas_segura` (solo sube, nunca baja). Si el texto no trae el email (p. ej. cita doc5 sin nombrarlo), `verificar_salida` (R_CANAL, T10) reemplaza la respuesta por la plantilla segura de escalamiento, que sí incluye el canal. Resultado: siempre `accion=escalar`, `canal` definido y el email en el texto.

**Excepción determinista.** Si el mensaje del cliente pregunta por los canales o medios de contacto (regex `_PREGUNTA_CANAL`: canal(es), contacto/contactar, comunicarme, teléfono, correo de soporte, horario de atención, dónde/cómo escribo/reporto, cómo hablo con soporte), la respuesta que informa el canal y cita doc5 conserva `responder` y `canal=None` (el canal se informa en el texto; no se deriva un caso). Si el LLM sugiere `escalar`, se mantiene.

## Alternativas consideradas
- Dejar la acción al LLM: es la causa del bug.
- Escalar toda cita de doc5 sin excepción: degrada las consultas informativas sobre canales.
- Clasificar la excepción con el LLM: añade otra llamada y no es determinista.

## Consecuencias
- Positivas: la acción nunca contradice el texto; las quejas coloquiales escalan por reglas, sin depender del LLM.
- Límites de la heurística de canal: es una regex acotada y se evalúa sobre el mensaje actual (no el historial). Un mensaje mixto («¿cuál es el canal? además el vendedor fue grosero») lo escalan igualmente las reglas de entrada; uno que mencione «contacto» sin ser pregunta de canal («perdí el contacto con el repartidor») no escala por remisión (se queda como `responder`), lo que es una degradación leve tolerada. Un FP inverso: «cómo reporto un problema con mi lavadora» cuenta como pregunta de canal.
- Costo extra de latencia por el clasificador activado, aceptado para cubrir redacciones que las reglas no ven.
