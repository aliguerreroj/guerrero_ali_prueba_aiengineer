"""Retriever híbrido (T07): relevancia sobre puntajes originales, RRF solo para ordenar.

Regla (ADR-004): un chunk es relevante si `bm25 > umbral_bm25` (y bm25 > 0) O
`coseno > umbral_semantico`. Si ninguno lo es, se devuelve `[]` (el orquestador
escala o pide un dato; nunca inventa). Solo los relevantes entran a la fusión
Reciprocal Rank Fusion, que únicamente ORDENA; después se corta a `top_k`.

Identidad: los chunks se identifican por la clave estable `clave_chunk`
`(doc_id, posicion)`, no por `id()`; así un VectorStore que devuelva copias
(p. ej. uno que serialice) no rompe la parte semántica.

Fallo del embedder: se degrada a solo BM25 y se registra un aviso (el fallo
seguro es perder la parte semántica, no tumbar la conversación). Si falla al
indexar, el retriever queda en modo solo léxico de forma permanente.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence

from pydantic import BaseModel, ConfigDict

from tiendahogar_agent.config import Settings
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import Chunk, clave_chunk
from tiendahogar_agent.puertos import Embedder, VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # constante habitual de Cormack et al. (2009); amortigua el peso de los primeros puestos


class FuenteChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    titulo: str | None = None


class ResultadoRecuperacion(BaseModel):
    """Chunk recuperado. `score` es el RRF (solo ordena); los originales sirven para trazas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: Chunk
    fuente: FuenteChunk
    score: float
    puntaje_bm25: float
    similitud: float | None = None  # None si no hubo parte semántica


def fusion_rrf(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> dict[Hashable, float]:
    """RRF: score(d) = suma sobre rankings de 1 / (k + posición), posición desde 1."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for pos, clave in enumerate(ranking, start=1):
            scores[clave] = scores.get(clave, 0.0) + 1.0 / (k + pos)
    return scores


def chunk_relevante(
    bm25: float, similitud: float | None, umbral_bm25: float, umbral_semantico: float
) -> bool:
    """Regla de ADR-004 sobre puntajes originales (la usan la calibración y los tests)."""
    return (bm25 > 0 and bm25 > umbral_bm25) or (
        similitud is not None and similitud > umbral_semantico
    )


def _fuente(chunk: Chunk) -> FuenteChunk:
    titulo = chunk.metadatos.get("titulo")
    return FuenteChunk(doc_id=chunk.doc_id, titulo=str(titulo) if titulo is not None else None)


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        indice: IndiceLexico,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        umbral_bm25: float,
        umbral_semantico: float,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k debe ser >= 1")
        self._chunks = list(chunks)
        self._por_clave: dict[tuple[str, int], Chunk] = {}
        for c in self._chunks:
            clave = clave_chunk(c)
            if clave in self._por_clave:
                raise ValueError(f"Clave de chunk duplicada: {clave}")
            self._por_clave[clave] = c
        self._indice = indice
        self._embedder = embedder
        self._store = store
        self.top_k = top_k
        self.umbral_bm25 = umbral_bm25
        self.umbral_semantico = umbral_semantico
        self._semantico = False
        if self._chunks:
            try:
                vectores = embedder.embed([c.texto for c in self._chunks])
                store.indexar(self._chunks, vectores)
                self._semantico = True
            except Exception:
                logger.warning("Embedder falló al indexar; se usa solo BM25", exc_info=True)

    def _similitudes(self, consulta: str) -> dict[tuple[str, int], float]:
        """Coseno de TODOS los chunks (sin umbral), indexado por clave estable."""
        if not self._semantico:
            return {}
        try:
            vector = self._embedder.embed([consulta])[0]
            pares = self._store.buscar(vector, k=len(self._chunks), umbral=-1.0)
        except Exception:
            logger.warning("Embedder falló en la consulta; se usa solo BM25", exc_info=True)
            return {}
        sims: dict[tuple[str, int], float] = {}
        for c, s in pares:
            try:
                clave = clave_chunk(c)
            except ValueError:
                clave = None
            if clave not in self._por_clave:
                logger.warning("El store devolvió un chunk desconocido (%s); se ignora", clave)
                continue
            sims[clave] = float(s)
        return sims

    def puntajes(self, consulta: str | None) -> list[tuple[Chunk, float, float | None]]:
        """(chunk, BM25, coseno|None) de TODOS los chunks, sin umbrales (para calibrar)."""
        if not consulta or not consulta.strip():
            return []
        bm25 = {clave_chunk(c): p for c, p in self._indice.puntuar(consulta)}
        sims = self._similitudes(consulta)
        return [
            (c, bm25.get(clave_chunk(c), 0.0), sims.get(clave_chunk(c))) for c in self._chunks
        ]

    def recuperar(self, consulta: str | None) -> list[ResultadoRecuperacion]:
        if not consulta or not consulta.strip() or not self._chunks:
            return []
        bm25 = {clave_chunk(c): p for c, p in self._indice.puntuar(consulta)}
        sims = self._similitudes(consulta)

        def lex_ok(c: Chunk) -> bool:
            p = bm25.get(clave_chunk(c), 0.0)
            return p > 0 and p > self.umbral_bm25

        def sem_ok(c: Chunk) -> bool:
            k = clave_chunk(c)
            return k in sims and sims[k] > self.umbral_semantico

        relevantes = [c for c in self._chunks if lex_ok(c) or sem_ok(c)]
        if not relevantes:
            return []

        # Cada ranking contiene solo los chunks relevantes por esa señal (puntaje original);
        # el desempate es el orden original (sort estable). El RRF solo ordena.
        r_lex = [clave_chunk(c) for c in sorted(filter(lex_ok, relevantes), key=lambda c: -bm25[clave_chunk(c)])]
        r_sem = [clave_chunk(c) for c in sorted(filter(sem_ok, relevantes), key=lambda c: -sims[clave_chunk(c)])]
        scores = fusion_rrf([r_lex, r_sem])

        ordenados = sorted(relevantes, key=lambda c: -scores[clave_chunk(c)])[: self.top_k]
        return [
            ResultadoRecuperacion(
                chunk=self._por_clave[clave_chunk(c)],
                fuente=_fuente(c),
                score=scores[clave_chunk(c)],
                puntaje_bm25=bm25.get(clave_chunk(c), 0.0),
                similitud=sims.get(clave_chunk(c)),
            )
            for c in ordenados
        ]


def construir_retriever(settings: Settings, chunks: list[Chunk]) -> Retriever:
    """Retriever real desde Settings. fastembed se importa solo al primer embed (perezoso)."""
    from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
    from tiendahogar_agent.adaptadores.fastembed_embedder import FastEmbedEmbedder

    return Retriever(
        chunks=chunks,
        indice=IndiceLexico(chunks),
        embedder=FastEmbedEmbedder(settings.embedding_model),
        store=InMemoryVectorStore(),
        top_k=settings.top_k,
        umbral_bm25=settings.umbral_bm25,
        umbral_semantico=settings.umbral_semantico,
    )
