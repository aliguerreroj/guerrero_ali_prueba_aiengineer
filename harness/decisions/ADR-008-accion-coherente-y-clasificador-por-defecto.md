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

## Acotación tras el eval real (T19) — 2026-10-04
**Qué cambia.** El apartado (c) ya no escala por el solo hecho de que el texto nombre `soporte@tiendahogar.example`. Ahora `Orquestador._entregar` sube la acción a `escalar` por remisión únicamente si la respuesta **cita `doc5`** (salvo que el cliente pregunte por los canales, excepción que se mantiene) o si el LLM sugirió `escalar` (lo cubre `accion_mas_segura`).

**Por qué.** El eval real con Haiku (`evals/resultados/2026-10-04-0047-*.md`) mostró que el modelo menciona el correo sin necesidad (casos vivo2-02, vivo2-06, t18-01, cuya acción esperada es `responder`) y la regla anterior convertía esas respuestas en escalamientos innecesarios.

**Nueva regla T10 `canal_innecesario`** (`guardrail_output.R_CANAL_INNECESARIO`): si la acción es `responder` o `pedir_dato` y el texto menciona el canal (sin tildes ni mayúsculas) sin `doc5` en `fuentes`, `verificar_salida` la marca en `reglas_fallidas`/`detalles` (fallo seguro para quien la llame directo). No aplica a `escalar`, ni si el llamador pasa `canal_solicitado=True` (el orquestador lo hace cuando `pregunta_por_canal(mensaje)`, para no castigar una respuesta que informa el canal sin citar doc5).

**Reintento único.** Calcado de `hecho_incorrecto` (ADR-009): el orquestador devuelve el error como resultado de la tool `responder` (con «no ejecutada» para las demás llamadas), pide reescribir sin mencionar a soporte y amplía el tope de iteraciones en uno; bandera propia e independiente de la de hecho, de modo que nunca hay más de una llamada extra por regla. Si fallan ambas reglas en la misma entrega se atiende primero el hecho; el canal se evalúa en la entrega siguiente (y no se gasta el reintento si el hecho persiste, porque saldría la respuesta segura igualmente). Queda en la traza (`reintentos`, `reglas_fallidas`) y en logs.

**Límite («persiste ⇒ escalar»).** Si el segundo intento sigue nombrando el canal sin `doc5`, se acepta el texto del LLM con acción `escalar` y el canal definido (lado seguro; ya contiene el correo, así que R_CANAL pasa) en vez de la plantilla `RESPUESTA_SEGURA`. Las demás reglas (cifras, compromisos, hechos) siguen aplicando. `accion_sugerida=escalar` del LLM no dispara la regla (la acción ya es `escalar`).

**Consecuencias.** Menos escalamientos espurios cuando el modelo menciona el correo de más; costo: una llamada LLM extra en esos turnos. La mención del correo sin `doc5` en una pregunta que no es de canales ya no se escala de inmediato, sino tras el reintento. Citar `doc5` sin nombrar el correo sigue dando la plantilla segura de escalamiento.
