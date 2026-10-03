"""Modo demostración (`LLM_PROVIDER=fake`): LLM determinista sin red, y retriever léxico.

`LLMDemo` implementa el puerto `LLMClient` con reglas fijas, solo para poder chatear en la CLI sin
API key. No es un modelo: no razona ni inventa; solo encadena las tools del orquestador.
- Si el mensaje menciona un pedido (`ORD-####`): llama `consultar_estado_pedido` y redacta con
  su resultado (éxito, no encontrado o formato inválido).
- Si no: llama `buscar_politicas` con el mensaje y responde con el comienzo del primer fragmento,
  citando su fuente. Sin fragmentos relevantes, dice que no tiene el dato y sugiere el canal humano
  (acción sugerida `escalar`).
Es independiente de `dobles` (que es solo para tests); `EmbedderConstante` evita descargar modelos
de embeddings: con umbral semántico imposible el retriever queda en la práctica solo en BM25.
"""

from __future__ import annotations

import html
import itertools
import json
import re
from typing import Any

from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.models import LlamadaTool, LLMResponse

_PEDIDO = re.compile(r"ORD\s*-?\s*\d{4}", re.IGNORECASE)
_DOCUMENTO = re.compile(r'<documento id="([^"]*)" fuente="[^"]*">\n(.*?)\n</documento>', re.DOTALL)
_MAX_CARACTERES_FRAGMENTO = 400
_RECORDATORIO_PREFIJO = "Recuerda entregar tu respuesta final"
UMBRAL_SEMANTICO_IMPOSIBLE = 2.0  # el coseno nunca supera 1


class EmbedderConstante:
    """Embedder sin modelo: vector fijo de dimensión 1 (con el umbral imposible desactiva lo semántico)."""

    def embed(self, textos: list[str]) -> list[list[float]]:
        return [[1.0] for _ in textos]


def _recortar(texto: str) -> str:
    """Primeras líneas del fragmento sin marcado, cortadas en un límite de frase o de palabra."""
    lineas = [re.sub(r"^[#>\-*\s]+", "", ln).replace("**", "").strip() for ln in texto.splitlines()]
    plano = " ".join(ln for ln in lineas if ln)
    if len(plano) <= _MAX_CARACTERES_FRAGMENTO:
        return plano
    corte = plano[:_MAX_CARACTERES_FRAGMENTO]
    fin = max(corte.rfind(". "), corte.rfind("; "))
    if fin > 100:
        return corte[: fin + 1]
    return corte[: corte.rfind(" ")] + "…"


class LLMDemo:
    """Ver docstring del módulo."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def completar(
        self,
        mensajes: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        usuario, resultado = self._situacion(mensajes)
        if resultado is None:
            pedido = _PEDIDO.search(usuario)
            if pedido:
                digitos = pedido.group(0)[-4:]
                return self._llamada("consultar_estado_pedido", {"order_id": f"ORD-{digitos}"})
            return self._llamada("buscar_politicas", {"consulta": usuario})
        nombre, contenido = resultado
        if nombre == "consultar_estado_pedido":
            return self._responder(*self._redactar_pedido(contenido))
        return self._responder(*self._redactar_politica(contenido))

    # ------------------------------------------------------------------ interno
    def _llamada(self, nombre: str, argumentos: dict[str, Any]) -> LLMResponse:
        ident = f"demo_{next(self._ids)}"
        return LLMResponse(llamadas_tools=[LlamadaTool(id=ident, nombre=nombre, argumentos=argumentos)])

    def _responder(self, texto: str, fuentes: list[str], sugerida: str | None) -> LLMResponse:
        args: dict[str, Any] = {"respuesta": texto, "fuentes": fuentes}
        if sugerida:
            args["accion_sugerida"] = sugerida
        return self._llamada("responder", args)

    @staticmethod
    def _situacion(mensajes: list[dict[str, Any]]) -> tuple[str, tuple[str, dict[str, Any]] | None]:
        """Último mensaje real del cliente y, si ya hubo, el último resultado de tool (nombre, json)."""
        indice = max(
            (
                i for i, m in enumerate(mensajes)
                if m.get("role") == "user"
                and not str(m.get("content", "")).startswith(_RECORDATORIO_PREFIJO)
            ),
            default=-1,
        )
        usuario = str(mensajes[indice]["content"]) if indice >= 0 else ""
        nombres: dict[str, str] = {}
        resultado: tuple[str, dict[str, Any]] | None = None
        for m in mensajes[indice + 1:]:
            for ll in m.get("tool_calls") or []:
                nombres[ll["id"]] = ll["name"]
            if m.get("role") == "tool":
                try:
                    datos = json.loads(m["content"])
                except (ValueError, TypeError):
                    datos = {}
                resultado = (nombres.get(m.get("tool_call_id"), ""), datos)
        return usuario, resultado

    @staticmethod
    def _redactar_pedido(datos: dict[str, Any]) -> tuple[str, list[str], str | None]:
        if datos.get("error") == "formato_invalido":
            return (
                ("El número de pedido debe tener el formato ORD-#### (por ejemplo, ORD-1001). "
                 "¿Me lo confirmas?"),
                [], "pedir_dato",
            )
        if datos.get("error"):
            return (
                (f"No encontré un pedido con el número {datos.get('order_id')}. "
                 "¿Puedes revisar que esté bien escrito?"),
                [], "pedir_dato",
            )
        texto = (
            f"Tu pedido {datos['order_id']} ({datos['producto']}) está en estado "
            f"«{datos['estado']}»."
        )
        if datos.get("entrega_estimada"):
            texto += f" La entrega estimada es en {datos['entrega_estimada']}."
        return texto, [], None

    @staticmethod
    def _redactar_politica(datos: dict[str, Any]) -> tuple[str, list[str], str | None]:
        bloque = _DOCUMENTO.search(datos.get("contexto", "")) if datos.get("fragmentos") else None
        if bloque is None:
            return (
                ("No encontré esa información en nuestros documentos, así que prefiero no "
                 f"inventarla. Escríbele a nuestro equipo en {CANAL_ESCALAMIENTO} y te ayudarán."),
                [], "escalar",
            )
        ident, texto = bloque.group(1), html.unescape(bloque.group(2))
        return f"Según nuestras políticas: {_recortar(texto)}", [html.unescape(ident)], None
