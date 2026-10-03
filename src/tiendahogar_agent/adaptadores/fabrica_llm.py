"""Fábrica de LLMClient según `settings.llm_provider`."""

from __future__ import annotations

from importlib import import_module

from tiendahogar_agent.config import Settings
from tiendahogar_agent.puertos import LLMClient


def _faltantes(settings: Settings) -> list[str]:
    """Nombres de los ajustes obligatorios que faltan para el proveedor elegido."""
    if settings.llm_provider == "anthropic":
        requeridos = {"anthropic_api_key": settings.anthropic_api_key}
    else:
        requeridos = {
            "azure_openai_api_key": settings.azure_openai_api_key,
            "azure_openai_endpoint": settings.azure_openai_endpoint,
            "azure_openai_deployment": settings.azure_openai_deployment,
        }
    return [nombre for nombre, valor in requeridos.items() if not valor]


def crear_llm(settings: Settings) -> LLMClient:
    """Devuelve el cliente LLM configurado. Si faltan credenciales, ValueError sin secretos."""
    if settings.llm_provider == "fake":
        # Import dinámico: ningún módulo de src importa `dobles` de forma estática.
        return import_module("tiendahogar_agent.dobles").FakeLLM()
    faltantes = _faltantes(settings)
    if faltantes:
        raise ValueError(
            f"Configuración incompleta para llm_provider='{settings.llm_provider}': "
            f"faltan {', '.join(faltantes)}."
        )
    if settings.llm_provider == "anthropic":
        from tiendahogar_agent.adaptadores.anthropic_llm import AnthropicLLM

        return AnthropicLLM(settings)
    from tiendahogar_agent.adaptadores.azure_openai_llm import AzureOpenAILLM

    return AzureOpenAILLM(settings)
