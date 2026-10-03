"""Pruebas del Retriever (T07). Unitarias deterministas; la integración usa el modelo real."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import Chunk, clave_chunk
from tiendahogar_agent.retriever import (
    RRF_K,
    ResultadoRecuperacion,
    Retriever,
    construir_retriever,
    fusion_rrf,
)

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
INALCANZABLE = 2.0  # el coseno nunca supera 1: desactiva la parte semántica


def mk(texto, doc_id, posicion=0):
    return Chunk(texto=texto, doc_id=doc_id, metadatos={"posicion": posicion})


@pytest.fixture(scope="module")
def chunks():
    return FileSystemDocumentSource(DOCS).cargar()


def hacer(chunks, embedder=None, top_k=3, umbral_bm25=0.5, umbral_semantico=INALCANZABLE):
    return Retriever(
        chunks,
        IndiceLexico(chunks),
        embedder or FakeEmbedder(),
        InMemoryVectorStore(),
        top_k=top_k,
        umbral_bm25=umbral_bm25,
        umbral_semantico=umbral_semantico,
    )


# --- RRF -----------------------------------------------------------------


def test_rrf_numerico():
    s = fusion_rrf([["a", "b", "c"], ["b", "a"]])
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["c"] == pytest.approx(1 / 63)
    assert RRF_K == 60
    assert fusion_rrf([]) == {}
    assert fusion_rrf([["x"]], k=0) == {"x": 1.0}


# --- BM25 con parte semántica desactivada ---------------------------------


@pytest.mark.parametrize(
    ("consulta", "doc"),
    [
        ("¿cuánto dura la garantía de mi licuadora?", "doc1"),
        ("quiero devolver un producto", "doc2"),
        ("¿hacen envíos a Miami?", "doc3"),
        ("un empleado me trató mal", "doc5"),
    ],
)
def test_documento_correcto_primero_sobre_bm25(chunks, consulta, doc):
    res = hacer(chunks).recuperar(consulta)
    assert res and res[0].chunk.doc_id == doc
    assert res[0].fuente.doc_id == doc and res[0].fuente.titulo
    assert res[0].puntaje_bm25 > 0


def test_resultados_ordenados_por_score_y_tipados(chunks):
    res = hacer(chunks, top_k=5).recuperar("garantía devolución envío")
    assert len(res) > 1
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
    assert all(isinstance(r, ResultadoRecuperacion) for r in res)
    with pytest.raises(ValidationError):
        res[0].score = 0  # frozen


def test_top_k_limita(chunks):
    consulta = "garantía devolución envío reembolso empleado"
    assert len(hacer(chunks, top_k=2).recuperar(consulta)) == 2
    assert len(hacer(chunks, top_k=1).recuperar(consulta)) == 1


def test_top_k_invalido(chunks):
    with pytest.raises(ValueError):
        hacer(chunks, top_k=0)


def test_umbral_bm25_configurable(chunks):
    consulta = "¿cuánto dura la garantía de mi licuadora?"
    assert hacer(chunks, umbral_bm25=0.5).recuperar(consulta)
    assert hacer(chunks, umbral_bm25=1000).recuperar(consulta) == []


def test_fuera_de_dominio_lista_vacia(chunks):
    assert hacer(chunks).recuperar("¿quién ganó el mundial?") == []


@pytest.mark.parametrize("consulta", ["", "   ", None, "¿?"])
def test_consulta_vacia(chunks, consulta):
    assert hacer(chunks).recuperar(consulta) == []


def test_sin_chunks_no_lanza():
    assert hacer([]).recuperar("garantía") == []


def test_chunks_de_texto_vacio_no_lanza():
    vacios = [mk("", "d1"), mk("   ", "d2")]
    assert hacer(vacios).recuperar("garantía") == []


def test_bm25_no_positivo_nunca_es_relevante():
    class IndiceNegativo:
        def __init__(self, ch):
            self.ch = ch

        def puntuar(self, consulta):
            return [(c, -1.0) for c in self.ch]

    ch = [mk("a", "d1"), mk("b", "d2")]
    r = Retriever(
        ch, IndiceNegativo(ch), FakeEmbedder(), InMemoryVectorStore(), 3, -5.0, INALCANZABLE
    )
    assert r.recuperar("a") == []


# --- identidad estable de chunks -------------------------------------------


def test_clave_chunk():
    assert clave_chunk(mk("x", "d1", 3)) == ("d1", 3)
    with pytest.raises(ValueError, match="posicion"):
        clave_chunk(Chunk(texto="x", doc_id="d1"))
    with pytest.raises(ValueError, match="posicion"):
        clave_chunk(Chunk(texto="x", doc_id="d1", metadatos={"posicion": "0"}))
    with pytest.raises(ValueError, match="posicion"):
        clave_chunk(Chunk(texto="x", doc_id="d1", metadatos={"posicion": True}))


def test_retriever_exige_posicion():
    with pytest.raises(ValueError, match="posicion"):
        hacer([Chunk(texto="a", doc_id="d1")])


def test_retriever_rechaza_claves_duplicadas():
    with pytest.raises(ValueError, match="duplicada"):
        hacer([mk("a", "d1", 0), mk("b", "d1", 0)])
    hacer([mk("a", "d1", 0), mk("b", "d1", 1), mk("c", "d2", 0)])  # distinto doc o posición: ok


class StoreConCopias:
    """Replica InMemoryVectorStore pero devuelve copias profundas de los chunks."""

    def __init__(self):
        self._interno = InMemoryVectorStore()

    def indexar(self, chunks, vectores):
        self._interno.indexar(chunks, vectores)

    def buscar(self, vector, k, umbral):
        return [(c.model_copy(deep=True), s) for c, s in self._interno.buscar(vector, k, umbral)]


def test_semantico_funciona_con_store_que_devuelve_copias():
    chs = [mk("alfa", "d1"), mk("beta", "d2")]
    emb = EmbedderPorTexto({"alfa": [1, 0], "beta": [0, 1], "zzz": [0.1, 1.0]})
    r = Retriever(chs, IndiceLexico(chs), emb, StoreConCopias(), 3, 0.5, 0.5)
    res = r.recuperar("zzz")
    assert [x.chunk.doc_id for x in res] == ["d2"]
    assert res[0].similitud is not None
    assert res[0].similitud == pytest.approx(0.995, abs=1e-3)
    assert res[0].chunk is chs[1]  # el chunk propio del retriever, no la copia del store


def test_clave_desconocida_del_store_se_ignora(caplog):
    class StoreAjeno(StoreConCopias):
        def buscar(self, vector, k, umbral):
            ajeno = mk("otro", "dX", 9)
            return [*super().buscar(vector, k, umbral), (ajeno, 1.0)]

    chs = [mk("alfa", "d1"), mk("beta", "d2")]
    emb = EmbedderPorTexto({"alfa": [1, 0], "beta": [0, 1], "zzz": [0.1, 1.0]})
    r = Retriever(chs, IndiceLexico(chs), emb, StoreAjeno(), 3, 0.5, 0.5)
    assert [x.chunk.doc_id for x in r.recuperar("zzz")] == ["d2"]
    assert "desconocido" in caplog.text


# --- parte semántica -----------------------------------------------------


class EmbedderPorTexto:
    """Vectores fijos: el texto decide a qué chunk se parece."""

    def __init__(self, tabla):
        self.tabla = tabla

    def embed(self, textos):
        return [self.tabla[t] for t in textos]


def test_semantico_rescata_chunk_sin_bm25():
    ch = [mk("alfa", "d1"), mk("beta", "d2")]
    emb = EmbedderPorTexto({"alfa": [1, 0], "beta": [0, 1], "zzz": [0.1, 1.0]})
    r = Retriever(ch, IndiceLexico(ch), emb, InMemoryVectorStore(), 3, 0.5, 0.5)
    res = r.recuperar("zzz")
    assert [x.chunk.doc_id for x in res] == ["d2"]
    assert res[0].puntaje_bm25 == 0
    assert res[0].similitud == pytest.approx(0.995, abs=1e-3)
    # El umbral semántico es configurable.
    r2 = Retriever(ch, IndiceLexico(ch), emb, InMemoryVectorStore(), 3, 0.5, 0.999)
    assert r2.recuperar("zzz") == []


def test_relevancia_se_decide_sobre_puntajes_originales_no_rrf():
    """Un chunk irrelevante en ambos puntajes no entra, aunque la fusión le daría score."""
    ch = [mk("alfa", "d1"), mk("beta", "d2")]
    emb = EmbedderPorTexto({"alfa": [1, 0], "beta": [0, 1], "alfa?": [1, 0]})
    r = Retriever(ch, IndiceLexico(ch), emb, InMemoryVectorStore(), 5, 0.1, 0.5)
    assert [x.chunk.doc_id for x in r.recuperar("alfa?")] == ["d1"]


class EmbedderRoto:
    def embed(self, textos):
        raise RuntimeError("sin red")


class EmbedderRotoEnConsulta:
    def __init__(self):
        self.n = 0

    def embed(self, textos):
        self.n += 1
        if self.n > 1:
            raise RuntimeError("falla")
        return [[1.0, 0.0] for _ in textos]


def test_fallo_del_embedder_al_indexar_degrada_a_bm25(chunks, caplog):
    r = hacer(chunks, embedder=EmbedderRoto(), umbral_semantico=0.0)
    res = r.recuperar("¿cuánto dura la garantía de mi licuadora?")
    assert res[0].chunk.doc_id == "doc1" and res[0].similitud is None
    assert "solo BM25" in caplog.text


def test_fallo_del_embedder_en_consulta_degrada(chunks):
    r = hacer(chunks, embedder=EmbedderRotoEnConsulta(), umbral_semantico=0.0)
    assert r.recuperar("quiero devolver un producto")[0].chunk.doc_id == "doc2"


# --- cableado ------------------------------------------------------------


def test_construir_retriever_desde_settings_es_perezoso(chunks, monkeypatch):
    """Construir no carga el modelo ni descarga nada: el motor solo se toca al indexar."""
    from tiendahogar_agent.adaptadores import fastembed_embedder as m

    def motor_falso(self):
        raise RuntimeError("sin modelo en tests")

    monkeypatch.setattr(m.FastEmbedEmbedder, "_motor", motor_falso)
    s = Settings(top_k=2, umbral_bm25=0.7, umbral_semantico=0.4)
    r = construir_retriever(s, chunks)
    assert (r.top_k, r.umbral_bm25, r.umbral_semantico) == (2, 0.7, 0.4)
    # Con el embedder caído degrada a BM25 y sigue respondiendo.
    assert r.recuperar("¿cuánto dura la garantía de mi licuadora?")[0].chunk.doc_id == "doc1"


# --- integración (modelo real; omitida por defecto) -------------------------

CONSULTAS = [
    ("¿cuánto dura la garantía de mi licuadora?", "doc1"),
    ("quiero devolver un producto", "doc2"),
    ("¿hacen envíos a Miami?", "doc3"),
    ("¿cuándo me devuelven el dinero?", "doc4"),
    ("un empleado me trató mal", "doc5"),
]
FUERA = "¿quién ganó el mundial?"


@pytest.mark.integration
def test_integracion_embedder_real(chunks):
    from tiendahogar_agent.adaptadores.fastembed_embedder import FastEmbedEmbedder

    s = Settings()
    emb = FastEmbedEmbedder(s.embedding_model)
    indice = IndiceLexico(chunks)
    store = InMemoryVectorStore()
    store.indexar(chunks, emb.embed([c.texto for c in chunks]))
    print()
    for consulta, _ in [*CONSULTAS, (FUERA, None)]:
        sims = store.buscar(emb.embed([consulta])[0], k=len(chunks), umbral=-1.0)
        print(f"Q: {consulta}")
        print("  coseno:", [(c.doc_id, round(v, 3)) for c, v in sims])
        print("  bm25  :", [(c.doc_id, round(v, 3)) for c, v in indice.puntuar(consulta)])

    real = construir_retriever(s, chunks)
    for consulta, doc in CONSULTAS:
        res = real.recuperar(consulta)
        assert res, consulta
        assert res[0].chunk.doc_id == doc, (consulta, [(r.chunk.doc_id, r.score) for r in res])
    assert real.recuperar(FUERA) == []
