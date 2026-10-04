# API HTTP del agente

API FastAPI sobre el mismo orquestador que usa la CLI (`cli.construir_orquestador`): misma
configuración (`LLM_PROVIDER`, `.env`, `settings.yaml`). Con `fake` (por defecto) no necesita red
ni API key.

## Arranque

Con el entorno activado y el paquete instalado (`python -m pip install -e ".[dev]"`):

```
python -m uvicorn tiendahogar_agent.api:app --port 8000
```

Opciones útiles: `--host 0.0.0.0` (aceptar conexiones externas; por defecto solo `127.0.0.1`) y
`--reload` (reinicio automático al editar código; solo en desarrollo).

El orquestador se construye en la primera petición a `/chat`, así que el arranque es inmediato.
La documentación interactiva (Swagger) está en <http://localhost:8000/docs>.
En Docker se usará el puerto 8080 (tarea T23).

## Endpoints

### `GET /health`

Responde `{"status": "ok"}`.

### `POST /chat`

Cuerpo JSON:

| Campo | Reglas |
|---|---|
| `conversation_id` | 1 a 64 caracteres entre letras ASCII, dígitos, `_`, `.`, `:` y `-` |
| `mensaje` | no vacío (ni solo espacios), máximo 2000 caracteres |

Respuesta 200: el modelo `AgentResponse` (`respuesta`, `accion` = `responder` | `escalar` |
`pedir_dato`, `fuentes`, `canal`, `trace_id`). Entradas inválidas devuelven 422. Un error
imprevisto devuelve un 500 genérico, sin traza ni datos personales (el detalle queda en el log).
Los casos que el dominio decide escalar llegan como 200 con `accion: "escalar"` y el canal.

Ejemplo en bash:

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "mensaje": "¿Dónde está mi pedido ORD-1001?"}'
```

Ejemplo en PowerShell (comillas simples por fuera; `curl.exe` para evitar el alias):

```
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{\"conversation_id\": \"demo-1\", \"mensaje\": \"¿Dónde está mi pedido ORD-1001?\"}'
```

Como alternativa en PowerShell:

```
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -ContentType "application/json; charset=utf-8" -Body (@{conversation_id="demo-1"; mensaje="¿Cuánto dura la garantía?"} | ConvertTo-Json)
```

## Historial de conversación

El servidor guarda el historial por `conversation_id` en memoria: se envía al orquestador en cada
turno, con un máximo de `MAX_MENSAJES_HISTORIAL` mensajes por conversación y de 1000
conversaciones (al superarlo se descarta la menos recientemente usada), con un lock para
seguridad entre hilos. Las conversaciones están aisladas entre sí.

## Notas de producción

- **Almacén externo.** El historial en memoria sirve para el prototipo: se pierde al reiniciar y no
  se comparte entre réplicas. En producción iría en un almacén externo (p. ej. Redis con TTL),
  detrás de la misma interfaz que `AlmacenHistorial`.
- **Autenticación y acceso.** El prototipo no autentica. En producción, la autenticación, la
  autorización y el control de acceso (zero-trust) irían en Apigee, delante de este servicio,
  junto con límites de tasa y cuotas.
- **Errores y privacidad.** El cliente nunca ve trazas ni PII; los logs y trazas enmascaran los
  datos personales.
