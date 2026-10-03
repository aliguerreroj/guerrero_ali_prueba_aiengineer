"""Puertos (interfaces) del dominio. No dependen de ningún SDK concreto."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tiendahogar_agent.models import Chunk, LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    """Cliente de modelo de lenguaje."""

    def completar(
        self,
        mensajes: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Envía mensajes (y definiciones de tools opcionales); devuelve texto y/o llamadas a tools."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Genera vectores a partir de textos."""

    def embed(self, textos: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Almacén vectorial con búsqueda por similitud."""

    def indexar(self, chunks: list[Chunk], vectores: list[list[float]]) -> None: ...

    def buscar(
        self, vector: list[float], k: int, umbral: float
    ) -> list[tuple[Chunk, float]]:
        """Devuelve hasta k pares (chunk, puntaje) con puntaje >= umbral, de mayor a menor."""
        ...


@runtime_checkable
class DocumentSource(Protocol):
    """Fuente de documentos ya troceados en chunks."""

    def cargar(self) -> list[Chunk]: ...


@runtime_checkable
class OrderRepository(Protocol):
    """Acceso al estado de pedidos."""

    def consultar_estado_pedido(self, order_id: str) -> dict[str, Any]: ...


@runtime_checkable
class EventBus(Protocol):
    """Publicación de eventos de observabilidad."""

    def publicar(self, evento: dict[str, Any]) -> None: ...


__all__ = [
    "DocumentSource",
    "Embedder",
    "EventBus",
    "LLMClient",
    "OrderRepository",
    "VectorStore",
]
