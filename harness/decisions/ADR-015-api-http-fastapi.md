# ADR-015 — API HTTP con FastAPI

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
Además de la CLI, el agente debe poder consumirse como servicio. La API debe reutilizar el mismo orquestador, validar la entrada y no filtrar detalles internos ni datos personales al cliente. Es un prototipo: se documentan sus límites en lugar de resolverlos todos.

## Decisión
Implementada en `src/tiendahogar_agent/api.py` y documentada en `docs/api.md`.
- **Endpoints**: `GET /health` (`{"status": "ok"}`) y `POST /chat` (modelo de respuesta `AgentResponse`). Arranque: `python -m uvicorn tiendahogar_agent.api:app --port 8000`.
- **Orquestador**: se construye con `cli.construir_orquestador` (misma configuración que la CLI) la primera vez que se usa `/chat`, no al importar el módulo; `crear_app` permite inyectar uno de prueba.
- **Validación** del cuerpo (`ChatRequest`, `extra="forbid"`): `conversation_id` de 1 a `MAX_LARGO_CONVERSATION_ID` (64) caracteres que cumplan `PATRON_CONVERSATION_ID` (`^[A-Za-z0-9_.:-]+$`); `mensaje` de como máximo `MAX_LARGO_MENSAJE` (2000) caracteres y no vacío ni solo espacios. Entradas inválidas devuelven 422.
- **Historial en memoria acotado** (`AlmacenHistorial`): como máximo `MAX_MENSAJES_HISTORIAL` mensajes por conversación (constante de `orquestador.py`) y `MAX_CONVERSACIONES` = 1000 conversaciones; al superarlo se descarta la menos recientemente usada. Protegido con un lock entre hilos; las conversaciones están aisladas.
- **Errores**: ante una excepción imprevista el cliente recibe un 500 con el texto genérico `MENSAJE_ERROR_INTERNO`, sin traza ni PII; al log solo va el tipo de la excepción. Lo que el dominio decide escalar sale como 200 con `accion: "escalar"` y el canal.
- **Idempotencia**: la API pasa `conversation_id` a `Orquestador.procesar`, que lo usa para la clave de idempotencia del evento `EscalationCreated` (ADR-010). Si un orquestador de prueba tiene la firma antigua sin ese parámetro, se llama sin él.

## Alternativas consideradas
- Flask o http.server: FastAPI da validación con pydantic y documentación interactiva (`/docs`) casi sin código.
- Historial en el cliente: obliga a cada consumidor a reenviarlo y permite manipularlo.
- Devolver el detalle de la excepción: útil al depurar, pero filtra información interna.

## Consecuencias
- Positivas: una sola lógica de dominio para CLI y API; entrada acotada; sin filtración de errores.
- Límites: el historial **no persiste** (se pierde al reiniciar) y **no es multi-proceso** ni se comparte entre réplicas; en producción iría en un almacén externo con TTL, detrás de la misma interfaz. No hay autenticación ni límites de tasa: en producción irían delante, en una pasarela de APIs.
