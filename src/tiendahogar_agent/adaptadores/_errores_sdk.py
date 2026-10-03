"""Traducción común de errores de los SDKs (anthropic/openai) a la jerarquía de dominio.

Ambos SDKs exponen las mismas clases base, por lo que una sola función sirve a los dos.
El mensaje de dominio NUNCA incluye el texto del error original (podría repetir la API key):
solo el proveedor, el tipo de error y el código HTTP.
"""

from __future__ import annotations

from types import ModuleType

from tiendahogar_agent.excepciones import (
    ErrorLLM,
    ErrorLLMAutenticacion,
    ErrorLLMConexion,
    ErrorLLMLimiteTasa,
    ErrorLLMServidor,
    ErrorLLMTimeout,
)


def traducir_error(error: Exception, sdk: ModuleType, proveedor: str) -> ErrorLLM:
    """Devuelve la excepción de dominio equivalente a `error` (error del SDK `sdk`)."""
    tipo = type(error).__name__
    # Orden importante: APITimeoutError es subclase de APIConnectionError.
    if isinstance(error, sdk.APITimeoutError):
        return ErrorLLMTimeout(f"{proveedor}: tiempo de espera agotado ({tipo})")
    if isinstance(error, sdk.APIConnectionError):
        return ErrorLLMConexion(f"{proveedor}: fallo de conexión ({tipo})")
    if isinstance(error, sdk.RateLimitError):
        return ErrorLLMLimiteTasa(f"{proveedor}: límite de tasa (HTTP 429)")
    if isinstance(error, sdk.APIStatusError):
        codigo = error.status_code
        if codigo in (401, 403):
            return ErrorLLMAutenticacion(f"{proveedor}: credenciales rechazadas (HTTP {codigo})")
        if codigo >= 500:
            return ErrorLLMServidor(f"{proveedor}: error del servidor (HTTP {codigo})")
        return ErrorLLM(f"{proveedor}: error de la API (HTTP {codigo})")
    return ErrorLLM(f"{proveedor}: error de la API ({tipo})")
