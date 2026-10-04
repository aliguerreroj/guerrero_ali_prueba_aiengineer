"""T22: evento `EscalationCreated` con clave de idempotencia determinista.

Todo con FakeLLM y buses de prueba, sin red ni API key.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.adaptadores.eventos_local import LocalEventBus
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeEventBus, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.eventos import (
    EscalationCreated,
    calcular_clave_idempotencia,
    construir_evento,
)
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.orquestador import Orquestador
from tiendahogar_agent.puertos import EventBus
from tiendahogar_agent.retriever import Retriever

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
CANAL = "soporte@tiendahogar.example"
MSG_LEGAL = "Voy a demandarlos y llamar a mi abogado por este asunto."
MSG_GOLPE = "Mi microondas llegó con un golpe, ¿qué hago?"
LIMPIO = "Lamento lo del golpe; cuéntame cómo llegó el empaque y vemos qué sigue."
_contador = itertools.count(1)


@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


def _responder(texto):
    return FakeLLM.llamada_tool(
        "responder", {"respuesta": texto, "fuentes": []}, id=f"c{next(_contador)}"
    )


def _orq(retriever, bus, *respuestas):
    llm = FakeLLM(list(respuestas))
    return Orquestador(
        llm, retriever, Settings(usar_clasificador_llm=False), event_bus=bus
    )


def _escalar_legal(retriever, bus, conversation_id=None, historial=None):
    orq = _orq(retriever, bus, FakeLLM.texto(f"Entiendo, escríbele a {CANAL}."))
    return orq.procesar(MSG_LEGAL, historial, conversation_id=conversation_id)


# ------------------------------------------------------------------ publicación
def test_escalar_publica_evento_con_campos_requeridos(retriever):
    bus = FakeEventBus()
    r = _escalar_legal(retriever, bus)
    assert r.accion == "escalar"
    assert len(bus.eventos) == 1
    ev = bus.eventos[0]
    assert ev["tipo"] == "EscalationCreated"
    assert ev["trace_id"] == r.trace_id
    assert ev["categoria"] == "legal"
    assert ev["canal"] == CANAL
    assert len(ev["idempotency_key"]) == 64
    assert ev["schema_version"] == 1
    EscalationCreated.model_validate(ev)


def test_fakeeventbus_cumple_el_puerto():
    assert isinstance(FakeEventBus(), EventBus)
    assert isinstance(LocalEventBus(), EventBus)


def test_no_se_publica_al_responder(retriever):
    bus = FakeEventBus()
    r = _orq(retriever, bus, _responder(LIMPIO)).procesar(MSG_GOLPE)
    assert r.accion == "responder"
    assert bus.eventos == []


def test_no_se_publica_al_pedir_dato(retriever):
    bus = FakeEventBus()
    r = _orq(retriever, bus).procesar("   ")
    assert r.accion == "pedir_dato"
    assert bus.eventos == []


def test_escalamiento_por_guardrail_de_salida_publica_una_sola_vez(retriever):
    bus = FakeEventBus()
    r = _orq(
        retriever, bus,
        _responder(f"Escríbele a {CANAL} y te ayudarán."),
        _responder(f"De todos modos escríbele a {CANAL}."),
    ).procesar(MSG_GOLPE)
    assert r.accion == "escalar"
    assert len(bus.eventos) == 1
    assert bus.eventos[0]["categoria"] == "fallo_seguro"
    assert bus.eventos[0]["trace_id"] == r.trace_id


def test_fallo_seguro_por_llm_caido_publica(retriever):
    bus = FakeEventBus()
    orq = _orq(retriever, bus)  # FakeLLM sin respuestas: RuntimeError -> fallo seguro
    r = orq.procesar(MSG_GOLPE)
    assert r.accion == "escalar"
    assert len(bus.eventos) == 1


def test_sin_bus_el_comportamiento_es_el_de_siempre(retriever):
    orq = Orquestador(
        FakeLLM([FakeLLM.texto(f"Escríbele a {CANAL}.")]), retriever,
        Settings(usar_clasificador_llm=False),
    )
    assert orq.procesar(MSG_LEGAL).accion == "escalar"


# ------------------------------------------------------------------ fallo del bus
class BusRoto:
    def publicar(self, evento):
        raise RuntimeError("broker caído: soporte@cliente.example 3001234567")


def test_fallo_del_bus_no_rompe_la_respuesta(retriever, caplog):
    caplog.set_level("WARNING")
    r = _escalar_legal(retriever, BusRoto())
    assert r.accion == "escalar" and r.canal == CANAL and CANAL in r.respuesta
    assert "no se pudo publicar" in caplog.text
    assert "3001234567" not in caplog.text and "cliente.example" not in caplog.text


# ------------------------------------------------------------------ clave
def test_clave_determinista_y_estable():
    kw = {
        "trace_id": "t1", "categoria": "legal", "canal": CANAL, "conversation_id": "c1",
        "turno": 2, "mensaje": "Voy a demandar",
    }
    clave = calcular_clave_idempotencia(**kw)
    assert clave == calcular_clave_idempotencia(**{**kw, "trace_id": "otro"})  # trace no entra
    assert clave == calcular_clave_idempotencia(**{**kw, "mensaje": "  VOY a   demandar "})
    assert len(clave) == 64
    assert clave.islower() and int(clave, 16) >= 0


@pytest.mark.parametrize("cambio", [
    {"conversation_id": "c2"}, {"turno": 3}, {"mensaje": "otro mensaje"},
    {"categoria": "reembolso_alto"}, {"canal": None},
])
def test_casos_distintos_generan_claves_distintas(cambio):
    kw = {
        "trace_id": "t1", "categoria": "legal", "canal": CANAL, "conversation_id": "c1",
        "turno": 2, "mensaje": "Voy a demandar",
    }
    assert calcular_clave_idempotencia(**kw) != calcular_clave_idempotencia(**{**kw, **cambio})


def test_sin_conversation_id_cada_turno_es_distinto():
    kw = {"categoria": "legal", "canal": CANAL, "mensaje": "x"}
    assert calcular_clave_idempotencia(trace_id="a", **kw) != calcular_clave_idempotencia(
        trace_id="b", **kw
    )


def test_reintento_del_mismo_turno_no_duplica(retriever):
    bus = LocalEventBus()
    r1 = _escalar_legal(retriever, bus, conversation_id="conv-1")
    r2 = _escalar_legal(retriever, bus, conversation_id="conv-1")
    assert r1.trace_id != r2.trace_id
    assert len(bus.eventos) == 1  # misma clave: el segundo se ignora


def test_otra_conversacion_o_turno_si_publica(retriever):
    bus = LocalEventBus()
    _escalar_legal(retriever, bus, conversation_id="conv-1")
    _escalar_legal(retriever, bus, conversation_id="conv-2")
    _escalar_legal(
        retriever, bus, conversation_id="conv-1",
        historial=[{"role": "user", "content": "hola"}, {"role": "assistant", "content": "hola"}],
    )
    assert len(bus.eventos) == 3
    assert len({e["idempotency_key"] for e in bus.eventos}) == 3


# ------------------------------------------------------------------ sin PII
def test_el_evento_no_contiene_pii_ni_texto_del_cliente(retriever):
    bus = FakeEventBus()
    mensaje = "Soy Ana Pérez, mi correo es ana.perez@correo.com y mi cel 3001234567; los demando"
    orq = _orq(retriever, bus, FakeLLM.texto(f"Entiendo, escríbele a {CANAL}."))
    orq.procesar(mensaje, conversation_id="conv-secreta-777")
    assert len(bus.eventos) == 1
    texto = json.dumps(bus.eventos[0], ensure_ascii=False)
    for secreto in ("Ana", "Pérez", "ana.perez", "3001234567", "demando", "conv-secreta-777"):
        assert secreto not in texto
    assert set(bus.eventos[0]) == {
        "tipo", "schema_version", "trace_id", "categoria", "canal", "idempotency_key",
        "ocurrido_en",
    }


def test_modelo_del_evento_es_estricto():
    ev = construir_evento(trace_id="t", categoria="legal", canal=CANAL)
    with pytest.raises(ValueError):
        EscalationCreated.model_validate({**ev.model_dump(), "texto_cliente": "hola"})


# ------------------------------------------------------------------ adaptador local
def _evento(clave="k1"):
    return {"tipo": "EscalationCreated", "idempotency_key": clave, "trace_id": "t"}


def test_adaptador_ignora_duplicados_en_memoria():
    bus = LocalEventBus()
    bus.publicar(_evento("a"))
    bus.publicar(_evento("a"))
    bus.publicar(_evento("b"))
    assert [e["idempotency_key"] for e in bus.eventos] == ["a", "b"]


def test_adaptador_persiste_y_no_republica_al_reabrir(tmp_path):
    ruta = tmp_path / "logs" / "eventos.jsonl"
    LocalEventBus(ruta).publicar(_evento("a"))
    reabierto = LocalEventBus(ruta)
    reabierto.publicar(_evento("a"))  # ya vista en el archivo
    reabierto.publicar(_evento("b"))
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["idempotency_key"] for x in lineas] == ["a", "b"]


def test_adaptador_tolera_lineas_corruptas(tmp_path):
    ruta = tmp_path / "e.jsonl"
    ruta.write_text('no es json\n{"idempotency_key": "a"}\n', encoding="utf-8")
    bus = LocalEventBus(ruta)
    bus.publicar(_evento("a"))
    bus.publicar(_evento("c"))
    assert len(ruta.read_text(encoding="utf-8").splitlines()) == 3


def test_construir_orquestador_conecta_el_bus_local(monkeypatch, tmp_path):
    from tiendahogar_agent.cli import construir_orquestador

    monkeypatch.setenv("TIENDAHOGAR_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    orq = construir_orquestador(Settings(llm_provider="fake"))
    r = orq.procesar(MSG_LEGAL, conversation_id="c-9")
    assert r.accion == "escalar"
    archivo = tmp_path / "logs" / "eventos_escalamiento.jsonl"
    assert len(archivo.read_text(encoding="utf-8").splitlines()) == 1


def test_linea_truncada_no_corrompe_el_siguiente_evento_ni_duplica(tmp_path):
    """Una última línea sin salto de línea no debe pegarse al evento nuevo (idempotencia al reabrir)."""
    from tiendahogar_agent.adaptadores.eventos_local import LocalEventBus

    ruta = tmp_path / "eventos.jsonl"
    ruta.write_text('{"idempotency_key": "A"}\n{"idempotency_key": "B', encoding="utf-8")
    LocalEventBus(ruta).publicar({"idempotency_key": "C"})
    reabierto = LocalEventBus(ruta)
    reabierto.publicar({"idempotency_key": "C"})  # duplicado: se ignora
    assert len(ruta.read_text(encoding="utf-8").splitlines()) == 3
    assert reabierto.eventos == []
