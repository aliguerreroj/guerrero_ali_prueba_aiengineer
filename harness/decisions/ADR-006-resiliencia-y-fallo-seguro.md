# ADR-006 — Resiliencia: reintentos del SDK, excepciones específicas y fallo seguro

- Estado: aceptada
- Fecha: 2026-10-02

## Contexto
El agente depende de un LLM remoto, de un retriever y de una tool de pedidos. Cualquiera puede fallar (timeout, red, límite de tasa, índice inconsistente, error interno). Un fallo no puede producir una respuesta inventada ni una excepción cruda al cliente, y los registros de error no pueden filtrar datos personales (correo, teléfono, tarjeta).

## Decisión
- **Reintentos delegados al SDK, sin bucle propio.** Los SDKs de LLM ya implementan reintentos con backoff exponencial y jitter (`max_retries`). `Settings` expone `timeout_llm_s`, `timeout_tool_s` y `max_reintentos_llm` (validados: timeouts en (0, tope], reintentos entre 0 y 5); T12 los pasa al cliente del SDK. El dominio llama una sola vez y trata cualquier error que le llegue como definitivo (los reintentos ya se agotaron). Así no hay `sleep` propios, ni duplicación de reintentos, ni tests lentos.
- **Excepciones específicas, nunca genéricas.** `excepciones.py` define `ErrorLLM` y subclases (`Timeout`, `Conexion`, `LimiteTasa`, `Autenticacion`, `Servidor`, `RespuestaInvalida`) como contrato: los adaptadores de T12 traducen las excepciones de cada SDK a esta jerarquía, de modo que el dominio no importa SDKs. Para el retriever se define `ErrorRecuperacion` y se capturan además `ValueError`, `KeyError` e `IndexError` (datos o índice inconsistentes). Está prohibido `except Exception`/`except:`; un error inesperado (p. ej. `TypeError`, que indica un bug) se propaga para no ocultarlo, y hay tests que lo demuestran.
- **Fallo seguro = escalar.** Ante error del LLM tras los reintentos, respuesta del LLM vacía (sin texto ni tools), `error_interno` de la tool de pedidos o fallo del retriever, se devuelve una `AgentResponse` con `accion="escalar"`, canal de soporte, sin fuentes y con un texto empático en tuteo, sin cifras ni compromisos (no dispara `verificar_salida`). Un único constructor (`respuesta_fallo_seguro`) la produce, con un `motivo` para trazas.
- **Registro con PII enmascarada.** Cada fallo se registra con `logging` (nivel ERROR) con motivo, tipo de excepción, acción aplicada y el mensaje pasado por `enmascarar_pii`. No se usa `exc_info` (el traceback incluiría el mensaje sin enmascarar).
- **API pequeña y testeable** en `resiliencia.py` (`llamar_llm_seguro`, `recuperar_seguro`, `revisar_resultado_tool`); pasa el timeout de `Settings` al LLM en cada llamada.

## Alternativas consideradas
- Bucle de reintentos propio con backoff: duplica lo que hace el SDK, multiplica latencia y complica los tests.
- Capturar `Exception` y escalar: oculta bugs y los convierte en escalamientos silenciosos.
- Responder con un mensaje de error técnico al usuario: peor experiencia; el fallo seguro del proyecto es escalar.

## Consecuencias
- Positivas: comportamiento uniforme ante fallos, sin PII en logs, tests deterministas sin red ni sleeps.
- Negativas: capturar `ValueError`/`KeyError`/`IndexError` en el retriever podría ocultar un bug de ese tipo dentro del retriever (se registra el tipo y se escala; se asume el costo). Si un adaptador de T12 deja pasar una excepción de SDK sin traducir, se propagará.
- El enmascarado de PII es el de T09 y hereda sus límites (p. ej. cifras adyacentes sin separador pueden leerse como un solo número).

## Fuera de alcance de T11
- La traducción real de excepciones de SDK a `ErrorLLM` y el paso de timeout/`max_retries` al cliente (T12).
- La orquestación que combina estas funciones con guardrails y tools (T14).
