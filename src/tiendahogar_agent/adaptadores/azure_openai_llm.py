"""Adaptador de LLMClient para Azure OpenAI (SDK `openai`, importado perezosamente)."""

from __future__ import annotations

import json
from typing import Any

from tiendahogar_agent.adaptadores._errores_sdk import traducir_error
from tiendahogar_agent.config import Settings
from tiendahogar_agent.excepciones import ErrorLLM, ErrorLLMRespuestaInvalida
from tiendahogar_agent.models import LlamadaTool, LLMResponse, UsoTokens

ESQUEMA_VACIO = {"type": "object", "properties": {}}


def _traducir_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Formato de dominio (name/description/parameters) -> function de chat.completions."""
    esquema = tool.get("parameters") or tool.get("input_schema") or ESQUEMA_VACIO
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": esquema,
        },
    }


class AzureOpenAILLM:
    """LLMClient sobre chat.completions de Azure OpenAI (el «modelo» es el deployment)."""

    def __init__(self, settings: Settings, cliente: Any | None = None) -> None:
        self._deployment = settings.azure_openai_deployment
        self._max_tokens = settings.llm_max_tokens
        import openai  # perezoso: el proveedor `fake` no requiere el SDK

        self._sdk = openai
        if cliente is None:
            clave = settings.azure_openai_api_key
            cliente = openai.AzureOpenAI(
                api_key=clave.get_secret_value() if clave else None,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                max_retries=settings.max_reintentos_llm,
                timeout=settings.timeout_llm_s,
            )
        self._cliente = cliente

    def completar(
        self,
        mensajes: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        parametros: dict[str, Any] = {
            "model": self._deployment,
            "messages": mensajes,
            "max_completion_tokens": self._max_tokens,
        }
        if tools:
            parametros["tools"] = [_traducir_tool(t) for t in tools]
        if timeout is not None:
            parametros["timeout"] = timeout
        try:
            respuesta = self._cliente.chat.completions.create(**parametros)
        except self._sdk.APIError as error:
            raise traducir_error(error, self._sdk, "azure_openai") from error
        return self._interpretar(respuesta)

    @staticmethod
    def _interpretar(respuesta: Any) -> LLMResponse:
        """Convierte la respuesta del SDK; si viene mal formada, ErrorLLMRespuestaInvalida."""
        try:
            mensaje = respuesta.choices[0].message
            llamadas: list[LlamadaTool] = []
            for llamada in mensaje.tool_calls or []:
                argumentos = json.loads(llamada.function.arguments or "{}")
                if not isinstance(argumentos, dict):
                    raise TypeError("los argumentos de la tool no son un objeto")
                llamadas.append(
                    LlamadaTool(id=llamada.id, nombre=llamada.function.name, argumentos=argumentos)
                )
            uso = UsoTokens(
                entrada=respuesta.usage.prompt_tokens, salida=respuesta.usage.completion_tokens
            )
            return LLMResponse(texto=mensaje.content or None, llamadas_tools=llamadas, uso=uso)
        except ErrorLLM:
            raise
        except Exception as error:
            raise ErrorLLMRespuestaInvalida(
                f"azure_openai: respuesta mal formada ({type(error).__name__})"
            ) from error
