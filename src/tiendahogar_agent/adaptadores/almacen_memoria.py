"""Adaptador del puerto VectorStore: almacén vectorial en memoria con similitud coseno."""

from __future__ import annotations

import math

from tiendahogar_agent.models import Chunk


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
