"""Pruebas de los puertos y sus dobles de prueba (sin red ni API key)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tiendahogar_agent.dobles import (
    FakeDocumentSource,
    FakeEmbedder,
    FakeEventBus,
    FakeLLM,
    FakeOrderRepository,
    InMemoryVectorStore,
)
from tiendahogar_agent.models import Chunk
from tiendahogar_agent.puertos import (
    DocumentSource,
    Embedder,
    EventBus,
    LLMClient,
    OrderRepository,
    VectorStore,
)

PAQUETE = Path(__file__).resolve().parents[1] / "src" / "tiendahogar_agent"
PROHIBIDOS = {"anthropic", "openai", "fastembed"}


def test_dobles_cumplen_protocolos():
    assert isinstance(FakeLLM([]), LLMClient)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(), VectorStore)
    assert isinstance(FakeDocumentSource([]), DocumentSource)
    assert isinstance(FakeOrderRepository({}), OrderRepository)
    assert isinstance(FakeEventBus(), EventBus)


def test_fake_llm_cola_y_registro():
    llm = FakeLLM([{"contenido": "a"}, {"contenido": "b"}])
    msgs = [{"role": "user", "content": "hola"}]
    assert llm.completar(msgs)["contenido"] == "a"
    assert llm.completar(msgs, tools=[{"name": "t"}], timeout=3.0)["contenido"] == "b"
    assert len(llm.llamadas) == 2
    assert llm.llamadas[1]["tools"] == [{"name": "t"}]
    assert llm.llamadas[1]["timeout"] == 3.0


def test_fake_llm_cola_agotada_lanza():
    with pytest.raises(RuntimeError):
        FakeLLM([]).completar([])


def test_fake_embedder_determinista():
    e = FakeEmbedder(dimension=8)
    v1 = e.embed(["hola", "mundo"])
    v2 = FakeEmbedder(dimension=8).embed(["hola", "mundo"])
    assert v1 == v2
    assert len(v1) == 2 and all(len(v) == 8 for v in v1)
    assert v1[0] != v1[1]


def test_vector_store_coseno_y_umbral():
    e = FakeEmbedder()
    chunks = [Chunk(texto=t, doc_id=f"d{i}") for i, t in enumerate(["garantia", "envios"])]
    store = InMemoryVectorStore()
    store.indexar(chunks, e.embed([c.texto for c in chunks]))
    consulta = e.embed(["garantia"])[0]
    res = store.buscar(consulta, k=2, umbral=-1.0)
    assert res[0][0].doc_id == "d0"
    assert abs(res[0][1] - 1.0) < 1e-9
    res2 = store.buscar(consulta, k=2, umbral=0.999)
    assert [c.doc_id for c, _ in res2] == ["d0"]
    assert len(store.buscar(consulta, k=1, umbral=-1.0)) == 1


def test_vector_store_vacio():
    assert InMemoryVectorStore().buscar([1.0, 0.0], k=3, umbral=0.0) == []


def test_document_source_fija():
    ch = [Chunk(texto="x", doc_id="d")]
    assert FakeDocumentSource(ch).cargar() == ch


def test_order_repository_dict():
    repo = FakeOrderRepository({"ORD-1": {"estado": "enviado"}})
    assert repo.consultar_estado_pedido("ORD-1") == {"estado": "enviado"}
    assert repo.consultar_estado_pedido("ORD-9")["estado"] == "no encontrado"


def test_event_bus_acumula():
    bus = FakeEventBus()
    bus.publicar({"tipo": "a"})
    bus.publicar({"tipo": "b"})
    assert bus.eventos == [{"tipo": "a"}, {"tipo": "b"}]


def test_puertos_sin_sdks_concretos():
    archivos = list((PAQUETE / "puertos").glob("*.py"))
    assert archivos
    for archivo in archivos:
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                raices = {a.name.split(".")[0] for a in nodo.names}
            elif isinstance(nodo, ast.ImportFrom):
                raices = {(nodo.module or "").split(".")[0]}
            else:
                continue
            assert not (raices & PROHIBIDOS), archivo
