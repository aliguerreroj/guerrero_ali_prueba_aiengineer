"""Pruebas de la recuperación léxica (BM25) y del embedder (T06). Sin red ni modelos."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.adaptadores.fastembed_embedder import (
    MODELO_POR_DEFECTO,
    FastEmbedEmbedder,
)
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import Chunk
from tiendahogar_agent.puertos import Embedder
from tiendahogar_agent.texto import normalizar

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture(scope="module")
def chunks():
    return FileSystemDocumentSource(DOCS).cargar()


@pytest.fixture(scope="module")
def indice(chunks):
    return IndiceLexico(chunks)


# --- normalización ---------------------------------------------------------


def test_normaliza_tildes_y_mayusculas():
    assert normalizar("GARANTÍA Garantia garantía") == 3 * normalizar("garantia")


def test_normaliza_enie_a_n():
    assert normalizar("Año") == normalizar("ano")


def test_stemming_agrupa_variantes():
    assert normalizar("licuadoras") == normalizar("licuadora")
    assert normalizar("envíos") == normalizar("envio")
    assert normalizar("devoluciones") == normalizar("devolucion")


def test_tokeniza_alfanumerico_y_signos():
    assert normalizar("¿Pedido ORD-1001, 30 días?") == normalizar("pedido ord 1001 30 dias")
    assert "1001" in normalizar("ORD-1001")


@pytest.mark.parametrize("entrada", ["", "   ", None, "¿?!"])
def test_entrada_vacia_o_none(entrada):
    assert normalizar(entrada) == []


def test_stopwords_se_descartan():
    assert normalizar("el de la") == []


# --- BM25 sobre los documentos reales -------------------------------------


@pytest.mark.parametrize(
    ("consulta", "doc"),
    [
        ("¿cuánto dura la garantía de mi licuadora?", "doc1"),
        ("quiero devolver un producto", "doc2"),
        ("¿hacen envíos a Miami?", "doc3"),
        ("un empleado me trató mal", "doc5"),
    ],
)
def test_bm25_documento_correcto_primero(indice, consulta, doc):
    ranking = indice.puntuar(consulta)
    assert ranking[0][0].doc_id == doc
    assert ranking[0][1] > 0


def test_bm25_brecha_lexica_documentada(indice):
    """«devuelven el dinero» no comparte raíz con «reembolsos»/«devuelto» de doc4.

    Limitación léxica real (no se fuerza con trucos): BM25 da 0 a todos y no hay
    ganador. Esta paráfrasis la cubre el embedder semántico y la fusión (T07).
    """
    assert all(p == 0 for _, p in indice.puntuar("¿cuándo me devuelven el dinero?"))
    # Con el término del documento sí acierta.
    assert indice.puntuar("¿cuándo me hacen el reembolso?")[0][0].doc_id == "doc4"


def test_bm25_ranking_completo_y_ordenado(indice, chunks):
    ranking = indice.puntuar("garantía licuadora")
    assert len(ranking) == len(chunks) == len(indice)
    puntajes = [p for _, p in ranking]
    assert puntajes == sorted(puntajes, reverse=True)
    assert all(isinstance(c, Chunk) for c, _ in ranking)


def test_bm25_fuera_de_dominio_puntaje_cero(indice):
    assert all(p == 0 for _, p in indice.puntuar("¿quién ganó el mundial?"))


@pytest.mark.parametrize("consulta", ["", None, "¿?"])
def test_bm25_consulta_vacia(indice, consulta):
    assert all(p == 0 for _, p in indice.puntuar(consulta))


def test_bm25_indice_vacio():
    assert IndiceLexico([]).puntuar("garantía") == []


# --- embedder ------------------------------------------------------------


class _TextEmbeddingFalso:
    def __init__(self):
        self.recibido = None

    def embed(self, textos):
        self.recibido = list(textos)
        return iter([[1, 2, 3] for _ in self.recibido])


def test_adaptador_fastembed_con_motor_inyectado():
    falso = _TextEmbeddingFalso()
    emb = FastEmbedEmbedder(text_embedding=falso)
    assert isinstance(emb, Embedder)
    assert emb.embed(["a", "b"]) == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert falso.recibido == ["a", "b"]
    assert emb.embed([]) == []
    assert emb.modelo == MODELO_POR_DEFECTO


def test_adaptador_no_carga_fastembed_al_construirse():
    emb = FastEmbedEmbedder("otro/modelo")
    assert emb.modelo == "otro/modelo" and emb._te is None


def test_fake_embedder_determinista_y_busqueda_end_to_end(chunks):
    emb = FakeEmbedder(dimension=32)
    assert emb.embed(["hola"]) == emb.embed(["hola"])
    store = InMemoryVectorStore()
    store.indexar(chunks, emb.embed([c.texto for c in chunks]))
    # La consulta idéntica a un chunk lo recupera primero (coseno = 1).
    res = store.buscar(emb.embed([chunks[2].texto])[0], k=3, umbral=-1.0)
    assert res[0][0] is chunks[2] and res[0][1] == pytest.approx(1.0)
    assert len(res) == 3


# --- configuración -------------------------------------------------------


def test_embedding_model_en_settings(monkeypatch):
    assert Settings().embedding_model == MODELO_POR_DEFECTO
    monkeypatch.setenv("EMBEDDING_MODEL", "otro/modelo")
    assert Settings().embedding_model == "otro/modelo"
    with pytest.raises(ValidationError):
        Settings(embedding_model="  ")
