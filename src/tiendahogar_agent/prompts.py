"""Prompts del sistema, contexto de documentos delimitado y esquema de la tool `responder`.

Funciones puras: leen plantillas versionadas en `plantillas/` (package data) y no tocan red.
El sistema decide la acción; el LLM solo redacta (ver `plantillas/sistema.md`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache
from importlib import resources
from typing import Any

from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.guardrail_output import FUENTE_PEDIDOS
from tiendahogar_agent.retriever import ResultadoRecuperacion

SIN_DOCUMENTOS = "No se recuperaron documentos relevantes para este mensaje."

NOMBRE_TOOL_BUSCAR = "buscar_politicas"
NOMBRE_TOOL_PEDIDO = "consultar_estado_pedido"
NOMBRE_TOOL_RESPONDER = "responder"
# Acciones que el LLM puede sugerir en `responder` (orden de menor a mayor seguridad en el orquestador).
ACCIONES_SUGERIBLES = ("pedir_dato", "escalar")

_MOTIVOS_ESCALAMIENTO = {
    "reembolso_alto": "una solicitud de reembolso de monto alto, que debe aprobar una persona supervisora",
    "queja_trato": "una queja sobre el trato recibido de parte de un empleado",
    "facturacion": "una disputa de facturación",
    "legal": "un tema legal",
    "manipulacion": "un mensaje que intenta cambiar las reglas del asistente",
}
_MOTIVO_GENERICO = "un caso que debe atender una persona del equipo"


@cache
def _leer_plantilla(nombre: str) -> str:
    recurso = resources.files("tiendahogar_agent").joinpath("plantillas", nombre)
    return recurso.read_text(encoding="utf-8")


def cargar_prompt_sistema(canal_humano: str = CANAL_ESCALAMIENTO) -> str:
    """Prompt del sistema con el canal humano insertado."""
    return _leer_plantilla("sistema.md").replace("{canal}", canal_humano)


def cargar_prompt_escalamiento(categoria: str, canal_humano: str = CANAL_ESCALAMIENTO) -> str:
    """Prompt para redactar el mensaje de escalamiento (sin tools) según la categoría."""
    motivo = _MOTIVOS_ESCALAMIENTO.get(categoria, _MOTIVO_GENERICO)
    return (
        _leer_plantilla("escalamiento.md").replace("{motivo}", motivo).replace("{canal}", canal_humano)
    )


def _neutralizar(texto: str) -> str:
    """Escapa `<` y `>` (incluidas variantes de ancho completo) para que no cierren la etiqueta."""
    return (
        texto.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("＜", "&lt;")
        .replace("＞", "&gt;")
    )


def _atributo_seguro(valor: str) -> str:
    return re.sub(r"[^\w.\-]", "_", valor)


def construir_contexto_documentos(
    resultados: Sequence[ResultadoRecuperacion], ids: Sequence[str] | None = None
) -> str:
    """Bloque con cada fragmento en `<documento id="docN" fuente="...">`; vacío si no hay.

    `ids` (opcional, uno por resultado) reemplaza el `docN` posicional; el orquestador pasa el
    `doc_id` real para que el id que cita el LLM coincida con el que verifica la salida.
    """
    if not resultados:
        cuerpo = SIN_DOCUMENTOS
    else:
        if ids is not None and len(ids) != len(resultados):
            raise ValueError("ids debe tener un elemento por resultado")
        bloques = [
            f'<documento id="{_atributo_seguro(ids[n] if ids is not None else f"doc{n + 1}")}" '
            f'fuente="{_atributo_seguro(r.fuente.doc_id)}">\n'
            f"{_neutralizar(r.chunk.texto)}\n</documento>"
            for n, r in enumerate(resultados)
        ]
        cuerpo = "\n\n".join(bloques)
    return _leer_plantilla("contexto_documentos.md").replace("{documentos}", cuerpo)


def definicion_tool_buscar_politicas() -> dict[str, Any]:
    """Tool de RAG: busca fragmentos de las políticas (garantía, devoluciones, envíos...)."""
    return {
        "name": NOMBRE_TOOL_BUSCAR,
        "description": (
            "Busca en los documentos de políticas de TiendaHogar (garantía, devoluciones, envíos, "
            "reembolsos, canales de contacto). Devuelve los fragmentos relevantes con su id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Qué quieres saber, en pocas palabras (p. ej. «garantía lavadora»).",
                },
            },
            "required": ["consulta"],
            "additionalProperties": False,
        },
    }


def definicion_tool_consultar_pedido() -> dict[str, Any]:
    """Tool de estado de pedido (firma real: consultar_estado_pedido(order_id: str))."""
    return {
        "name": NOMBRE_TOOL_PEDIDO,
        "description": (
            "Consulta el estado de un pedido por su número. Úsala solo cuando el cliente "
            "haya dado el número; si no lo tienes, pídeselo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Número del pedido, con formato ORD-#### (p. ej. ORD-1001).",
                },
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    }


def definicion_tool_responder() -> dict[str, Any]:
    """Tool final de salida estructurada, en formato de dominio {name, description, parameters}."""
    return {
        "name": NOMBRE_TOOL_RESPONDER,
        "description": (
            "Entrega la respuesta final al cliente. Úsala una sola vez, al terminar, "
            "con un mensaje breve (2 a 4 frases) basado solo en los documentos y las tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "respuesta": {
                    "type": "string",
                    "description": "Mensaje para el cliente, en español, tuteo y 2 a 4 frases.",
                },
                "fuentes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Identificadores de los documentos usados (p. ej. doc1). Si usaste "
                        f"consultar_estado_pedido, incluye también «{FUENTE_PEDIDOS}»."
                    ),
                },
                "accion_sugerida": {
                    "type": "string",
                    "enum": list(ACCIONES_SUGERIBLES),
                    "description": (
                        "Opcional. pedir_dato si tu mensaje pide al cliente un dato que falta; "
                        "escalar si no hay sustento en los documentos o las herramientas "
                        "(incluye el canal humano en el mensaje). Omítelo si respondes normal."
                    ),
                },
            },
            "required": ["respuesta"],
            "additionalProperties": False,
        },
    }
