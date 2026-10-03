"""Formato de dominio de los mensajes con tools (independiente de cualquier SDK).

Los mensajes del puerto `LLMClient` siguen siendo `list[dict]`. Roles:
- ``{"role": "assistant", "content": str | None, "tool_calls": [{"id", "name", "arguments": dict}]}``
  (sin la clave ``tool_calls`` si no hubo llamadas);
- ``{"role": "tool", "tool_call_id": str, "content": str, "es_error": bool}``: el contenido ya
  viene serializado por el orquestador (``json.dumps(..., ensure_ascii=False)``).
Cada adaptador traduce este formato al de su SDK.
"""

from __future__ import annotations

from typing import Any

from tiendahogar_agent.excepciones import ErrorHistorialMensajes
from tiendahogar_agent.models import LlamadaTool, LLMResponse


def mensaje_asistente(texto: str | None, llamadas: list[LlamadaTool]) -> dict[str, Any]:
    """Mensaje del asistente, con ``tool_calls`` solo si hay llamadas."""
    mensaje: dict[str, Any] = {"role": "assistant", "content": texto}
    if llamadas:
        mensaje["tool_calls"] = [
            {"id": ll.id, "name": ll.nombre, "arguments": dict(ll.argumentos)} for ll in llamadas
        ]
    return mensaje


def mensaje_resultado_tool(
    tool_call_id: str, contenido: str, es_error: bool = False
) -> dict[str, Any]:
    """Resultado de una tool (``contenido`` ya serializado a str)."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": contenido,
        "es_error": es_error,
    }


def mensaje_desde_respuesta(respuesta: LLMResponse) -> dict[str, Any]:
    """Reinyecta la respuesta del LLM en el historial."""
    return mensaje_asistente(respuesta.texto, respuesta.llamadas_tools)


def validar_historial(mensajes: list[dict[str, Any]]) -> None:
    """Falla rápido si el historial con tools es incoherente.

    Reglas: todo ``tool_call`` del asistente necesita su resultado antes de cualquier mensaje
    que no sea ``tool``; todo resultado debe referir un id pendiente (no vacío, no repetido);
    un mensaje del asistente sin ``tool_calls`` no puede tener el contenido vacío (los SDK lo
    rechazan y reinyectarlo indica un bug del orquestador).
    """
    pendientes: set[str] = set()
    vistos: set[str] = set()
    for i, m in enumerate(mensajes):
        rol = m.get("role")
        if rol == "tool":
            ident = m.get("tool_call_id")
            if not isinstance(ident, str) or not ident:
                raise ErrorHistorialMensajes(f"mensaje {i}: tool_call_id vacío")
            if ident not in pendientes:
                raise ErrorHistorialMensajes(
                    f"mensaje {i}: resultado de tool sin llamada previa del asistente"
                )
            if not isinstance(m.get("content"), str):
                raise ErrorHistorialMensajes(f"mensaje {i}: el contenido de tool debe ser str")
            pendientes.discard(ident)
            continue
        if pendientes:
            raise ErrorHistorialMensajes(
                f"mensaje {i}: faltan resultados de tool para {len(pendientes)} llamada(s)"
            )
        if rol == "assistant":
            contenido = m.get("content")
            if not m.get("tool_calls") and not (isinstance(contenido, str) and contenido.strip()):
                raise ErrorHistorialMensajes(f"mensaje {i}: asistente vacío sin llamadas de tool")
            for ll in m.get("tool_calls") or []:
                ident = ll.get("id")
                if not isinstance(ident, str) or not ident:
                    raise ErrorHistorialMensajes(f"mensaje {i}: llamada de tool con id vacío")
                if ident in vistos:
                    raise ErrorHistorialMensajes(f"mensaje {i}: id de llamada repetido")
                if not isinstance(ll.get("arguments"), dict) or not ll.get("name"):
                    raise ErrorHistorialMensajes(f"mensaje {i}: llamada de tool mal formada")
                vistos.add(ident)
                pendientes.add(ident)
    if pendientes:
        raise ErrorHistorialMensajes(
            f"historial termina con {len(pendientes)} llamada(s) de tool sin resultado"
        )
