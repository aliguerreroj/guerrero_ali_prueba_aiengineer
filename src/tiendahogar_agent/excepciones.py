"""Excepciones de dominio (contrato para los adaptadores).

Los adaptadores (T12) traducen las excepciones de cada SDK a esta jerarquía, de modo
que el dominio nunca importa un SDK. Los reintentos con backoff los hace el propio SDK
(`max_retries` del cliente); estas excepciones se lanzan cuando ya se agotaron.

Mapeo esperado en T12:
- timeout del SDK            -> ErrorLLMTimeout
- fallo de red/conexión      -> ErrorLLMConexion
- HTTP 429                   -> ErrorLLMLimiteTasa
- HTTP 401/403               -> ErrorLLMAutenticacion
- HTTP 5xx                   -> ErrorLLMServidor
- respuesta mal formada      -> ErrorLLMRespuestaInvalida
- cualquier otro error del SDK conocido -> ErrorLLM
"""

from __future__ import annotations


class ErrorLLM(Exception):
    """Fallo del modelo de lenguaje (base de la jerarquía)."""


class ErrorLLMTimeout(ErrorLLM):
    """La llamada superó el tiempo máximo."""


class ErrorLLMConexion(ErrorLLM):
    """No se pudo conectar con el proveedor."""


class ErrorLLMLimiteTasa(ErrorLLM):
    """El proveedor rechazó la llamada por límite de tasa."""


class ErrorLLMAutenticacion(ErrorLLM):
    """Credenciales inválidas o sin permiso."""


class ErrorLLMServidor(ErrorLLM):
    """Error 5xx del proveedor."""


class ErrorLLMRespuestaInvalida(ErrorLLM):
    """La respuesta del proveedor no se pudo interpretar."""


class ErrorRecuperacion(Exception):
    """Fallo al recuperar fragmentos de los documentos."""


class ErrorHistorialMensajes(ValueError):
    """Historial de mensajes mal formado (error de programación, no del proveedor).

    Se lanza antes de llamar al LLM: p. ej. un resultado de tool sin su llamada previa, un
    `tool_call_id` vacío o una llamada del asistente sin resultado. No es un `ErrorLLM`
    a propósito: no debe ocultarse como fallo seguro. Los mensajes nunca incluyen el contenido.
    """
