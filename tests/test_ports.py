"""Pruebas de los puertos y sus dobles de prueba (sin red ni API key)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore, _coseno
from tiendahogar_agent.dobles import (
    FakeDocumentSource,
    FakeEmbedder,
    FakeEventBus,
    FakeLLM,
    FakeOrderRepository,
)
from tiendahogar_agent.models import Chunk, LlamadaTool, LLMResponse, UsoTokens
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
    llm = FakeLLM([FakeLLM.texto("a", 10, 2), FakeLLM.texto("b")])
    msgs = [{"role": "user", "content": "hola"}]
    r1 = llm.completar(msgs)
    assert isinstance(r1, LLMResponse)
    assert r1.texto == "a" and r1.llamadas_tools == []
    assert r1.uso.entrada == 10 and r1.uso.salida == 2
    assert llm.completar(msgs, tools=[{"name": "t"}], timeout=3.0).texto == "b"
    assert len(llm.llamadas) == 2
    assert llm.llamadas[1]["tools"] == [{"name": "t"}]
    assert llm.llamadas[1]["timeout"] == 3.0


def test_fake_llm_llamada_tool():
    llm = FakeLLM()
    llm.encolar(FakeLLM.llamada_tool("consultar_estado_pedido", {"order_id": "ORD-1"}, id="c9"))
    r = llm.completar([{"role": "user", "content": "x"}])
    assert r.texto is None
    assert len(r.llamadas_tools) == 1
    lt = r.llamadas_tools[0]
    assert (lt.id, lt.nombre, lt.argumentos) == (
        "c9",
        "consultar_estado_pedido",
        {"order_id": "ORD-1"},
    )


def test_llm_response_mixta_y_extra_prohibido():
    r = LLMResponse(texto="t", llamadas_tools=[LlamadaTool(id="1", nombre="n")])
    assert r.llamadas_tools[0].argumentos == {}
    with pytest.raises(ValidationError):
        LLMResponse(texto="t", extra=1)
    with pytest.raises(ValidationError):
        UsoTokens(entrada=-1)


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


def test_coseno_dimensiones_distintas():
    with pytest.raises(ValueError):
        _coseno([1.0, 0.0], [1.0])


def _store_basico():
    store = InMemoryVectorStore()
    chunks = [Chunk(texto="a", doc_id="a"), Chunk(texto="b", doc_id="b")]
    store.indexar(chunks, [[1.0, 0.0], [0.0, 1.0]])
    return store


@pytest.mark.parametrize("k", [0, -1])
def test_buscar_k_cero_o_negativo(k):
    assert _store_basico().buscar([1.0, 0.0], k=k, umbral=-1.0) == []


def test_buscar_k_mayor_que_chunks():
    assert len(_store_basico().buscar([1.0, 0.0], k=50, umbral=-1.0)) == 2


def test_buscar_umbrales_extremos():
    s = _store_basico()
    assert len(s.buscar([1.0, 0.0], k=5, umbral=0.0)) == 2
    assert [c.doc_id for c, _ in s.buscar([1.0, 0.0], k=5, umbral=1.0)] == ["a"]
    assert s.buscar([1.0, 0.0], k=5, umbral=1.5) == []
    assert s.buscar([1.0, 0.0], k=5, umbral=float("nan")) == []


def test_buscar_vector_nulo():
    s = _store_basico()
    assert s.buscar([0.0, 0.0], k=5, umbral=0.1) == []
    assert len(s.buscar([0.0, 0.0], k=5, umbral=0.0)) == 2


def test_buscar_dimension_distinta_lanza():
    with pytest.raises(ValueError):
        _store_basico().buscar([1.0, 0.0, 0.0], k=1, umbral=0.0)


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


def test_ningun_modulo_de_src_importa_dobles():
    for archivo in PAQUETE.rglob("*.py"):
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                nombres = [a.name for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom):
                modulo = nodo.module or ""
                nombres = [modulo] + [f"{modulo}.{a.name}" for a in nodo.names]
            else:
                continue
            assert "tiendahogar_agent.dobles" not in nombres, archivo
