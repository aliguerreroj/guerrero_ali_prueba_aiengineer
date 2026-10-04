"""Pruebas de la API FastAPI (T20) con TestClient y FakeLLM: sin red ni API key."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.api import (
    MAX_LARGO_MENSAJE,
    AlmacenHistorial,
    crear_app,
)
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import AgentResponse
from tiendahogar_agent.orquestador import MAX_MENSAJES_HISTORIAL, Orquestador
from tiendahogar_agent.retriever import Retriever

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("LLM_PROVIDER", "USAR_CLASIFICADOR_LLM", "MAX_ITERACIONES_LLM"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


def _llm_tool(nombre, argumentos, n):
    return FakeLLM.llamada_tool(nombre, argumentos, id=f"api{n}")


def _cliente(retriever, *respuestas, **kw):
    llm = FakeLLM(list(respuestas))
    orq = Orquestador(llm, retriever, Settings(usar_clasificador_llm=False))
    return TestClient(crear_app(orq, **kw)), llm


class OrqEco:
    """Orquestador falso: registra el historial recibido."""

    def __init__(self):
        self.recibido = []

    def procesar(self, mensaje, historial=None):
        self.recibido.append((mensaje, historial))
        return AgentResponse(
            respuesta=f"eco:{mensaje}", accion="responder", trace_id=f"t{len(self.recibido)}"
        )


def _post(cliente, conv, mensaje):
    return cliente.post("/chat", json={"conversation_id": conv, "mensaje": mensaje})


def test_health():
    cliente = TestClient(crear_app(OrqEco()))
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_docs_responde_200():
    assert TestClient(crear_app(OrqEco())).get("/docs").status_code == 200


def test_chat_feliz_con_fakellm(retriever):
    cliente, _ = _cliente(
        retriever,
        _llm_tool("buscar_politicas", {"consulta": "garantía lavadora"}, 1),
        _llm_tool("responder", {"respuesta": "Te cubre la garantía legal.", "fuentes": ["doc1"]}, 2),
    )
    r = _post(cliente, "c1", "¿Cuánto dura la garantía de mi lavadora?")
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"respuesta", "accion", "fuentes", "canal", "trace_id"}
    assert cuerpo["accion"] == "responder" and cuerpo["fuentes"] == ["doc1"]
    AgentResponse.model_validate(cuerpo)


def test_chat_pedido(retriever):
    cliente, llm = _cliente(
        retriever,
        _llm_tool("consultar_estado_pedido", {"order_id": "ORD-1001"}, 1),
        _llm_tool("responder", {"respuesta": "Tu pedido ORD-1001 está En tránsito."}, 2),
    )
    r = _post(cliente, "c1", "¿Dónde está mi pedido ORD-1001?")
    assert r.status_code == 200 and r.json()["accion"] == "responder"
    assert len(llm.llamadas) == 2


def test_chat_escalamiento(retriever):
    cliente, _ = _cliente(retriever, FakeLLM.texto("Te paso con una persona del equipo."))
    r = _post(cliente, "c1", "Quiero un reembolso de $800 por mi lavadora")
    assert r.status_code == 200
    assert r.json()["accion"] == "escalar" and r.json()["canal"] == CANAL_ESCALAMIENTO


def test_historial_se_acumula_en_la_misma_conversacion():
    orq = OrqEco()
    cliente = TestClient(crear_app(orq))
    _post(cliente, "a", "hola")
    _post(cliente, "a", "segunda")
    assert orq.recibido[0][1] == []
    assert orq.recibido[1][1] == [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "eco:hola"},
    ]


def test_historial_aislado_entre_conversaciones():
    orq = OrqEco()
    cliente = TestClient(crear_app(orq))
    _post(cliente, "a", "hola A")
    _post(cliente, "b", "hola B")
    assert orq.recibido[1][1] == []


def test_historial_llega_al_llm(retriever):
    cliente, llm = _cliente(
        retriever,
        FakeLLM.llamada_tool("responder", {"respuesta": "Hola, ¿en qué te ayudo?"}, id="h1"),
        FakeLLM.llamada_tool("responder", {"respuesta": "Claro."}, id="h2"),
    )
    _post(cliente, "a", "Hola buenas")
    _post(cliente, "a", "Gracias")
    contenidos = [m.get("content") for m in llm.llamadas[1]["mensajes"]]
    assert "Hola buenas" in contenidos


def test_limite_de_turnos():
    orq = OrqEco()
    cliente = TestClient(crear_app(orq))
    for i in range(MAX_MENSAJES_HISTORIAL):  # más turnos que el tope
        _post(cliente, "a", f"m{i}")
    historial = orq.recibido[-1][1]
    assert len(historial) <= MAX_MENSAJES_HISTORIAL
    assert historial[0]["role"] == "user"
    assert historial[-1] == {
        "role": "assistant", "content": f"eco:m{MAX_MENSAJES_HISTORIAL - 2}"
    }


def test_limite_de_conversaciones_descarta_la_mas_antigua():
    orq = OrqEco()
    almacen = AlmacenHistorial(max_conversaciones=2)
    cliente = TestClient(crear_app(orq, almacen=almacen))
    _post(cliente, "a", "uno")
    _post(cliente, "b", "dos")
    _post(cliente, "c", "tres")  # expulsa a «a»
    assert len(almacen) == 2
    _post(cliente, "a", "otra vez")
    assert orq.recibido[-1][1] == []
    _post(cliente, "c", "sigo")
    assert orq.recibido[-1][1] != []


@pytest.mark.parametrize("mensaje", ["", "   ", "\n\t "])
def test_mensaje_vacio_422(mensaje):
    r = _post(TestClient(crear_app(OrqEco())), "a", mensaje)
    assert r.status_code == 422


def test_mensaje_demasiado_largo_422():
    cliente = TestClient(crear_app(OrqEco()))
    assert _post(cliente, "a", "x" * MAX_LARGO_MENSAJE).status_code == 200
    assert _post(cliente, "a", "x" * (MAX_LARGO_MENSAJE + 1)).status_code == 422


@pytest.mark.parametrize("conv", ["", "a" * 65, "con espacio", "../x", "ñandú"])
def test_conversation_id_invalido_422(conv):
    assert _post(TestClient(crear_app(OrqEco())), conv, "hola").status_code == 422


def test_campos_faltantes_o_extra_422():
    cliente = TestClient(crear_app(OrqEco()))
    assert cliente.post("/chat", json={"mensaje": "hola"}).status_code == 422
    assert cliente.post(
        "/chat", json={"conversation_id": "a", "mensaje": "hola", "x": 1}
    ).status_code == 422


def test_error_inesperado_500_generico():
    class Roto:
        def procesar(self, mensaje, historial=None):
            raise RuntimeError("secreto ana@correo.com Traceback")

    r = _post(TestClient(crear_app(Roto())), "a", "hola")
    assert r.status_code == 500
    assert "ana@correo.com" not in r.text and "Traceback" not in r.text
    assert "RuntimeError" not in r.text


def test_app_por_defecto_no_construye_el_orquestador_al_importar():
    from tiendahogar_agent import api

    assert TestClient(api.app).get("/health").status_code == 200
