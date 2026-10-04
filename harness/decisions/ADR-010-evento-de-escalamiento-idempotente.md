# ADR-010 — Evento de escalamiento con clave de idempotencia

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
Cuando el agente escala a un humano, otros sistemas (tickets, CRM, notificaciones) deben enterarse. Los reintentos (del cliente, del cliente HTTP, del propio productor) no deben crear tickets duplicados, y el evento no debe filtrar PII.

## Decisión
1. **Evento `EscalationCreated`** (`eventos.py`, pydantic estricto e inmutable): `tipo`, `schema_version` (1), `trace_id`, `categoria`, `canal`, `idempotency_key`, `ocurrido_en`. Sin texto del cliente, sin `conversation_id` en claro, sin PII.
2. **Publicación por el puerto `EventBus`** desde `Orquestador.procesar`, una sola vez por turno cuya acción final sea `escalar`, sea cual sea el camino (regla determinista, clasificador, verificación de salida, fallo seguro). Si no se inyecta bus, nada cambia. Publicar es mejor esfuerzo: si el bus falla se registra un WARNING (solo trace_id y tipo de excepción) y el cliente recibe igual su respuesta de escalamiento. La categoría es la del guardrail de entrada; si el escalamiento nace de la verificación de salida o de un fallo, es `fallo_seguro`.
3. **Clave de idempotencia** = sha256 de `v1 | conversation_id | n.º de mensajes del cliente en el historial | huella(mensaje normalizado) | categoria | canal`. Se excluyen a propósito `trace_id` (uuid nuevo por intento) y la hora, para que un reintento del mismo turno dé la misma clave. Entran la conversación, el turno y el mensaje para que casos distintos no colisionen (si no, el segundo cliente con el mismo texto perdería su escalamiento). Sin `conversation_id` (CLI, evals) la base es el `trace_id`: cada turno es un caso distinto, nunca se pierde un escalamiento. La API pasa su `conversation_id` (el orquestador es retrocompatible: parámetro opcional).
4. **Adaptador local** `LocalEventBus` (`adaptadores/eventos_local.py`): memoria más JSONL opcional (`logs/eventos_escalamiento.jsonl`, ignorado por git, ruta configurable por `TIENDAHOGAR_LOGS_DIR`). Ignora claves ya vistas, también las del archivo al reabrirlo (tolera líneas corruptas). Lock de hilo dentro del proceso. `cli.construir_orquestador` lo conecta por defecto (la API lo usa a través de esa función).

## Cómo se reemplazaría en producción (Kafka o Azure Event Hubs)
- **Productor**: adaptador de `EventBus` con `enable.idempotence=true` y `acks=all` (Kafka) o reintentos con la misma clave (Event Hubs). Esquema versionado (`schema_version`, Schema Registry con Avro/JSON Schema y compatibilidad hacia atrás); sin PII (el evento ya no la lleva).
- **Clave de partición**: el hash de `conversation_id` (o la `idempotency_key`), para conservar el orden por conversación y que los duplicados caigan en la misma partición.
- **Consumidor idempotente**: deduplica por `idempotency_key` (tabla con restricción única, o `SET NX` con TTL en Redis) antes de crear el ticket; entrega «al menos una vez» más deduplicación equivale a efecto único. Commit de offsets tras procesar.
- **Outbox**: si la decisión de escalar y el registro en base deben ser atómicos, se escribe el evento en una tabla outbox en la misma transacción y un relay lo publica; hoy, al ser mejor esfuerzo y sin base propia, no hace falta.
- **DLQ**: eventos que fallan tras N reintentos van a una cola de mensajes muertos con alerta; el fallo del bus nunca bloquea la respuesta al cliente.
- Un Schema Registry y un consumidor de DLQ quedan como trabajo de producción.

## Alternativas consideradas
- Clave basada solo en `trace_id`: no deduplica reintentos (uuid nuevo por intento).
- Incluir el texto del cliente en el evento: viola la regla de no PII; solo su huella alimenta el hash.
- Publicar desde cada punto de escalamiento: riesgo de duplicados y de olvidar rutas; se centraliza al final del turno.
- Archivo con lock entre procesos: complejidad innecesaria para un prototipo.

## Consecuencias
- Positivas: escalamientos observables sin acoplar el dominio a un broker; reintentos del mismo turno no duplican; sin PII.
- Límites: la huella de un mensaje corto y predecible es adivinable por fuerza bruta (se publica solo la clave, que mezcla conversación y turno); si el cliente repite el mismo mensaje después de que el historial creció se trata como caso nuevo (lado seguro); el bus local no coordina varios procesos; `fallo_seguro` agrupa la verificación de salida y los fallos técnicos.
