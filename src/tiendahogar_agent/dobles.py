"""Dobles de prueba deterministas para los puertos (sin red ni API key)."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from tiendahogar_agent.models import Chunk, LlamadaTool, LLMResponse, UsoTokens


class FakeLLM:
    """LLM guionizado: devuelve LLMResponse en orden y registra las llamadas."""

    def __init__(self, respuestas: list[LLMResponse] | None = None) -> None:
        self._cola: list[LLMResponse] = list(respuestas or [])
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

    def encolar(self, respuesta: LLMResponse) -> None:
        self._cola.append(respuesta)

    def completar(
        self,
        mensajes: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        self.llamadas.append({"mensajes": mensajes, "tools": tools, "timeout": timeout})
        if not self._cola:
            raise RuntimeError("FakeLLM: no quedan respuestas guionizadas")
        return self._cola.pop(0)


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


def _coseno(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimensiones distintas: {len(a)} != {len(b)}")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


class InMemoryVectorStore:
    """Almacén vectorial en memoria con similitud coseno."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def indexar(self, chunks: list[Chunk], vectores: list[list[float]]) -> None:
        if len(chunks) != len(vectores):
            raise ValueError("chunks y vectores deben tener la misma longitud")
        self._items.extend(zip(chunks, vectores, strict=True))

    def buscar(
        self, vector: list[float], k: int, umbral: float
    ) -> list[tuple[Chunk, float]]:
        puntuados = [(c, _coseno(vector, v)) for c, v in self._items]
        validos = [p for p in puntuados if p[1] >= umbral]
        validos.sort(key=lambda p: p[1], reverse=True)
        return validos[: max(k, 0)]


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
