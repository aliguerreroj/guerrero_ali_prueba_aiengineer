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
from tiendahogar_agent.retriever import ResultadoRecuperacion

SIN_DOCUMENTOS = "No hay documentos relevantes para esta consulta."


@cache
def _leer_plantilla(nombre: str) -> str:
    recurso = resources.files("tiendahogar_agent").joinpath("plantillas", nombre)
    return recurso.read_text(encoding="utf-8")


def cargar_prompt_sistema(canal_humano: str = CANAL_ESCALAMIENTO) -> str:
    """Prompt del sistema con el canal humano insertado."""
    return _leer_plantilla("sistema.md").replace("{canal}", canal_humano)


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


def construir_contexto_documentos(resultados: Sequence[ResultadoRecuperacion]) -> str:
    """Bloque con cada fragmento en `<documento id="docN" fuente="...">`; vacío si no hay."""
    if not resultados:
        cuerpo = SIN_DOCUMENTOS
    else:
        bloques = [
            f'<documento id="doc{n}" fuente="{_atributo_seguro(r.fuente.doc_id)}">\n'
            f"{_neutralizar(r.chunk.texto)}\n</documento>"
            for n, r in enumerate(resultados, start=1)
        ]
        cuerpo = "\n\n".join(bloques)
    return _leer_plantilla("contexto_documentos.md").replace("{documentos}", cuerpo)


def definicion_tool_responder() -> dict[str, Any]:
    """Tool final de salida estructurada, en formato de dominio {name, description, parameters}."""
    return {
        "name": "responder",
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
                    "description": "Identificadores de los documentos usados (p. ej. doc1).",
                },
            },
            "required": ["respuesta"],
            "additionalProperties": False,
        },
    }
