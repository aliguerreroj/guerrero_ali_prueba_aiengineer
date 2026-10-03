"""Adaptador de LLMClient para Anthropic (SDK oficial `anthropic`, importado perezosamente)."""

from __future__ import annotations

from typing import Any

from tiendahogar_agent.adaptadores._errores_sdk import traducir_error
from tiendahogar_agent.config import Settings
from tiendahogar_agent.excepciones import ErrorLLM, ErrorLLMRespuestaInvalida
from tiendahogar_agent.mensajes import validar_historial
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


def _traducir_mensajes(mensajes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mensajes de dominio (sin system) -> formato Anthropic.

    El asistente con tools pasa a bloques `text`/`tool_use`; los resultados `tool` consecutivos
    se fusionan en UN mensaje `user` con bloques `tool_result` (lo exige la API).
    """
    salida: list[dict[str, Any]] = []
    en_resultados = False  # el último mensaje de salida agrupa resultados de tool
    for m in mensajes:
        rol = m.get("role")
        if rol == "tool":
            bloque: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            if m.get("es_error"):
                bloque["is_error"] = True
            if en_resultados:
                salida[-1]["content"].append(bloque)
            else:
                salida.append({"role": "user", "content": [bloque]})
                en_resultados = True
            continue
        en_resultados = False
        if rol == "assistant" and m.get("tool_calls"):
            bloques: list[dict[str, Any]] = []
            # Anthropic rechaza bloques text vacíos o de solo espacios.
            if isinstance(m.get("content"), str) and m["content"].strip():
                bloques.append({"type": "text", "text": m["content"]})
            for ll in m["tool_calls"]:
                bloques.append(
                    {
                        "type": "tool_use",
                        "id": ll["id"],
                        "name": ll["name"],
                        "input": ll["arguments"],
                    }
                )
            salida.append({"role": "assistant", "content": bloques})
        else:
            salida.append(m)
    return salida


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
        max_tokens: int | None = None,
    ) -> LLMResponse:
        validar_historial(mensajes)
        sistema = "\n\n".join(str(m["content"]) for m in mensajes if m.get("role") == "system")
        conversacion = _traducir_mensajes([m for m in mensajes if m.get("role") != "system"])
        parametros: dict[str, Any] = {
            "model": self._modelo,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
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
