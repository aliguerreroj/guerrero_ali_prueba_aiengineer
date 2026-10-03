"""Dobles de prueba deterministas para los puertos (sin red ni API key)."""

from __future__ import annotations

import hashlib
from typing import Any

from tiendahogar_agent.mensajes import validar_historial
from tiendahogar_agent.models import Chunk, LlamadaTool, LLMResponse, UsoTokens


class FakeLLM:
    """LLM guionizado: devuelve LLMResponse en orden (o lanza la excepción encolada)."""

    def __init__(self, respuestas: list[LLMResponse | BaseException] | None = None) -> None:
        self._cola: list[LLMResponse | BaseException] = list(respuestas or [])
        self.llamadas: list[dict[str, Any]] = []

    @staticmethod
    def texto(
        contenido: str, tokens_entrada: int = 0, tokens_salida: int = 0
    ) -> LLMResponse:
        """Helper: respuesta de solo texto."""
        return LLMResponse(
            texto=contenido, uso=UsoTokens(entrada=tokens_entrada, salida=tokens_salida)
        )

    @staticmethod
    def llamada_tool(
        nombre: str,
        argumentos: dict[str, Any] | None = None,
        id: str = "call_1",
        tokens_entrada: int = 0,
        tokens_salida: int = 0,
    ) -> LLMResponse:
        """Helper: respuesta con una llamada a tool."""
        return LLMResponse(
            llamadas_tools=[LlamadaTool(id=id, nombre=nombre, argumentos=argumentos or {})],
            uso=UsoTokens(entrada=tokens_entrada, salida=tokens_salida),
        )

    @staticmethod
    def vacia() -> LLMResponse:
        """Helper: respuesta sin texto ni llamadas a tools."""
        return LLMResponse(texto=None)

    def encolar(self, respuesta: LLMResponse) -> None:
        self._cola.append(respuesta)

    def encolar_error(self, error: BaseException) -> None:
        """Encola una excepción que se lanzará en la siguiente llamada."""
        self._cola.append(error)

    def encolar_vacia(self) -> None:
        self._cola.append(self.vacia())

    def completar(
        self,
        mensajes: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        validar_historial(mensajes)  # detecta historiales con tools mal formados
        self.llamadas.append(
            {"mensajes": mensajes, "tools": tools, "timeout": timeout, "max_tokens": max_tokens}
        )
        if not self._cola:
            raise RuntimeError("FakeLLM: no quedan respuestas guionizadas")
        siguiente = self._cola.pop(0)
        if isinstance(siguiente, BaseException):
            raise siguiente
        return siguiente


class FakeEmbedder:
    """Vector determinista derivado de SHA-256 (independiente de PYTHONHASHSEED)."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def _vector(self, texto: str) -> list[float]:
        valores: list[float] = []
        contador = 0
        while len(valores) < self.dimension:
            digest = hashlib.sha256(f"{contador}:{texto}".encode()).digest()
            valores.extend(b / 255.0 - 0.5 for b in digest)
            contador += 1
        return valores[: self.dimension]

    def embed(self, textos: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in textos]


class FakeDocumentSource:
    """Fuente con una lista fija de chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)

    def cargar(self) -> list[Chunk]:
        return list(self._chunks)


class FakeOrderRepository:
    """Repositorio en memoria inyectable (no es la tabla mock oficial)."""

    def __init__(self, pedidos: dict[str, dict[str, Any]]) -> None:
        self._pedidos = dict(pedidos)

    def consultar_estado_pedido(self, order_id: str) -> dict[str, Any]:
        if order_id in self._pedidos:
            return dict(self._pedidos[order_id])
        return {"order_id": order_id, "estado": "no encontrado"}


class FakeEventBus:
    """Bus que acumula los eventos publicados."""

    def __init__(self) -> None:
        self.eventos: list[dict[str, Any]] = []

    def publicar(self, evento: dict[str, Any]) -> None:
        self.eventos.append(evento)
