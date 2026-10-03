"""Índice léxico BM25 sobre chunks.

Se usa `BM25Okapi` con `epsilon`: con un corpus de pocos documentos, un término
presente en la mitad o más de los chunks tiene IDF negativo; `rank_bm25` lo
sustituye por `epsilon * IDF_promedio` (positivo), así que ningún término resta puntaje.

`puntuar` devuelve el ranking COMPLETO con los puntajes BM25 originales, sin
umbral ni top_k: la fusión y los cortes se aplican después (T07).
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from tiendahogar_agent.models import Chunk
from tiendahogar_agent.texto import normalizar

EPSILON = 0.25


class IndiceLexico:
    def __init__(self, chunks: list[Chunk], epsilon: float = EPSILON) -> None:
        self._chunks = list(chunks)
        corpus = [normalizar(c.texto) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus, epsilon=epsilon) if any(corpus) else None

    def __len__(self) -> int:
        return len(self._chunks)

    def puntuar(self, consulta: str | None) -> list[tuple[Chunk, float]]:
        """Todos los chunks con su puntaje BM25, de mayor a menor (empates: orden original)."""
        tokens = normalizar(consulta)
        if self._bm25 is None or not tokens:
            puntajes = [0.0] * len(self._chunks)
        else:
            puntajes = [float(p) for p in self._bm25.get_scores(tokens)]
        pares = list(zip(self._chunks, puntajes, strict=True))
        pares.sort(key=lambda p: p[1], reverse=True)  # sort estable
        return pares
