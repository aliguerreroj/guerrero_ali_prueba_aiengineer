"""Adaptador de LLMClient para Anthropic (SDK oficial `anthropic`, importado perezosamente)."""

from __future__ import annotations

from typing import Any

from tiendahogar_agent.adaptadores._errores_sdk import traducir_error
from tiendahogar_agent.config import Settings
from tiendahogar_agent.excepciones import ErrorLLM, ErrorLLMRespuestaInvalida
from tiendahogar_agent.models import LlamadaTool, LLMResponse, UsoTokens

ESQUEMA_VACIO = {"type": "object", "properties": {}}


def _traducir_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Formato de dominio (name/description/parameters) -> formato Anthropic."""
    esquema = tool.get("input_schema") or tool.get("parameters") or ESQUEMA_VACIO
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": esquema,
    }


class AnthropicLLM:
    """LLMClient sobre la API de mensajes de Anthropic."""

    def __init__(self, settings: Settings, cliente: Any | None = None) -> None:
        self._modelo = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        import anthropic  # perezoso: el proveedor `fake` no requiere el SDK

        self._sdk = anthropic
        if cliente is None:
            clave = settings.anthropic_api_key
            cliente = anthropic.Anthropic(
                api_key=clave.get_secret_value() if clave else None,
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
        sistema = "\n\n".join(str(m["content"]) for m in mensajes if m.get("role") == "system")
        conversacion = [m for m in mensajes if m.get("role") != "system"]
        parametros: dict[str, Any] = {
            "model": self._modelo,
            "max_tokens": self._max_tokens,
            "messages": conversacion,
        }
        if sistema:
            parametros["system"] = sistema
        if tools:
            parametros["tools"] = [_traducir_tool(t) for t in tools]
        if timeout is not None:
            parametros["timeout"] = timeout
        try:
            respuesta = self._cliente.messages.create(**parametros)
        except self._sdk.APIError as error:
            raise traducir_error(error, self._sdk, "anthropic") from error
        return self._interpretar(respuesta)

    @staticmethod
    def _interpretar(respuesta: Any) -> LLMResponse:
        """Convierte la respuesta del SDK; si viene mal formada, ErrorLLMRespuestaInvalida."""
        try:
            textos: list[str] = []
            llamadas: list[LlamadaTool] = []
            for bloque in respuesta.content:
                if bloque.type == "text":
                    textos.append(bloque.text)
                elif bloque.type == "tool_use":
                    if not isinstance(bloque.input, dict):
                        raise TypeError("input de tool_use no es un objeto")
                    llamadas.append(
                        LlamadaTool(id=bloque.id, nombre=bloque.name, argumentos=bloque.input)
                    )
            uso = UsoTokens(
                entrada=respuesta.usage.input_tokens, salida=respuesta.usage.output_tokens
            )
            return LLMResponse(texto="".join(textos) or None, llamadas_tools=llamadas, uso=uso)
        except ErrorLLM:
            raise
        except Exception as error:
            raise ErrorLLMRespuestaInvalida(
                f"anthropic: respuesta mal formada ({type(error).__name__})"
            ) from error
