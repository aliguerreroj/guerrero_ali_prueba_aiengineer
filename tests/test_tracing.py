"""Pruebas de las trazas JSONL (T17) con FakeLLM: deterministas, sin red ni API key."""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.adaptadores.traza_jsonl import JsonlTraceSink
from tiendahogar_agent.clasificador import NOMBRE_TOOL as TOOL_CLASIFICAR
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM, FakeTraceSink
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.orquestador import Orquestador
from tiendahogar_agent.puertos import TraceSink
from tiendahogar_agent.retriever import Retriever

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "data" / "docs"
_contador = itertools.count(1)
CLAVES = {
    "trace_id", "accion", "categoria", "regla", "documentos", "tools", "reglas_fallidas",
    "reintentos", "tokens_entrada", "tokens_salida", "costo_usd", "modelo", "latencia_ms",
}
PII = ["ana.perez@correo.com", "+57 300 123 4567", "4111 1111 1111 1111"]


@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


def _ll(nombre, argumentos, **tokens):
    return FakeLLM.llamada_tool(nombre, argumentos, id=f"t{next(_contador)}", **tokens)


def _orq(retriever, *respuestas, sink=None, clasificador=False, **kw):
    llm = FakeLLM(list(respuestas))
    s = Settings(usar_clasificador_llm=clasificador, **kw)
    return Orquestador(llm, retriever, s, trace_sink=sink), llm


def test_sink_cumple_el_puerto():
    assert isinstance(FakeTraceSink(), TraceSink)
    assert isinstance(JsonlTraceSink(Path(".")), TraceSink)


def test_una_traza_por_turno_con_todas_las_claves(retriever):
    sink = FakeTraceSink()
    orq, _ = _orq(
        retriever,
        _ll("buscar_politicas", {"consulta": "garantía lavadora"}, tokens_entrada=10, tokens_salida=2),
        _ll("responder", {"respuesta": "Las lavadoras tienen 12 meses de garantía.", "fuentes": ["doc1"]},
            tokens_entrada=20, tokens_salida=5),
        sink=sink,
    )
    r = orq.procesar("¿Cuánto dura la garantía de mi lavadora?")
    assert len(sink.trazas) == 1
    t = sink.trazas[0]
    assert CLAVES <= set(t)
    assert t["trace_id"] == r.trace_id and t["accion"] == "responder"
    assert t["reintentos"] == 0 and t["reglas_fallidas"] == []
    assert t["latencia_ms"] >= 0 and t["modelo"] == "claude-haiku-4-5-20251001"
    origenes = {d["origen"] for d in t["documentos"]}
    assert origenes == {"rag_automatico", "buscar_politicas"}
    for d in t["documentos"]:
        assert {"doc_id", "puntaje_bm25", "similitud", "score"} <= set(d)
        assert d["similitud"] is None or isinstance(d["similitud"], float)
        assert isinstance(d["puntaje_bm25"], float) and isinstance(d["score"], float)
    assert [x["nombre"] for x in t["tools"]] == ["buscar_politicas", "responder"]
    assert "respuesta" not in t["tools"][1]["argumentos"]
    json.dumps(t)


def test_escalamiento_por_guardrail_registra_categoria_y_regla(retriever):
    sink = FakeTraceSink()
    texto = f"Entiendo tu situación. Escríbele a {CANAL_ESCALAMIENTO} y te ayudarán."
    orq, _ = _orq(retriever, FakeLLM.texto(texto, 7, 3), sink=sink)
    r = orq.procesar("Voy a demandar a la tienda con mi abogado")
    assert r.accion == "escalar"
    t = sink.trazas[0]
    assert t["accion"] == "escalar" and t["categoria"] == "legal" and t["regla"]
    assert t["tokens_entrada"] == 7 and t["tokens_salida"] == 3


def test_tokens_suman_clasificador_bucle_y_reintento_y_costo(retriever):
    sink = FakeTraceSink()
    orq, llm = _orq(
        retriever,
        _ll(TOOL_CLASIFICAR, {"intencion": "politica"}, tokens_entrada=100, tokens_salida=10),
        _ll("responder", {"respuesta": "Tu licuadora tiene 12 meses de garantía."},
            tokens_entrada=1000, tokens_salida=50),
        _ll("responder", {"respuesta": "Tu licuadora, como electrodoméstico pequeño, tiene 6 meses de garantía."},
            tokens_entrada=2000, tokens_salida=60),
        sink=sink, clasificador=True,
        precio_entrada_por_millon=1.0, precio_salida_por_millon=5.0,
    )
    orq.procesar("¿Cuánto dura la garantía de mi licuadora?")
    assert len(llm.llamadas) == 3
    t = sink.trazas[0]
    assert t["tokens_entrada"] == 3100 and t["tokens_salida"] == 120
    assert t["reintentos"] == 1
    assert "hecho_incorrecto" in t["reglas_fallidas"]
    assert t["costo_usd"] == pytest.approx(3100 / 1e6 * 1.0 + 120 / 1e6 * 5.0)
    assert t["accion"] == "responder"


def test_costo_es_estimacion_si_el_modelo_no_es_el_de_los_precios(retriever):
    sink = FakeTraceSink()
    orq, _ = _orq(retriever, _ll("responder", {"respuesta": "Hola, ¿en qué te ayudo?"}),
                  sink=sink, llm_model="otro-modelo")
    orq.procesar("hola")
    assert sink.trazas[0]["costo_es_estimacion"] is True and sink.trazas[0]["modelo"] == "otro-modelo"


def test_pii_ausente_de_toda_la_linea_jsonl(retriever, tmp_path):
    sink = JsonlTraceSink(tmp_path / "logs")
    consulta = f"garantía {PII[0]} {PII[1]} {PII[2]}"
    orq, _ = _orq(
        retriever,
        _ll("buscar_politicas", {"consulta": consulta}),
        _ll("responder", {"respuesta": f"Gracias {PII[0]}, tienes 12 meses de garantía.", "fuentes": ["doc1"]}),
        sink=sink,
    )
    orq.procesar(f"Mi correo es {PII[0]}, tel {PII[1]}, tarjeta {PII[2]}: garantía lavadora")
    lineas = (tmp_path / "logs" / "trazas.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1
    registro = json.loads(lineas[0])
    for dato in PII:
        assert dato not in lineas[0]
        assert dato.replace(" ", "") not in lineas[0]
    assert "[CORREO]" in registro["tools"][0]["argumentos"]["consulta"]
    assert "Mi correo es" not in lineas[0]


def test_fallo_del_sink_no_rompe_ni_altera_el_turno(retriever, caplog):
    class Roto:
        def registrar(self, traza):
            raise OSError(f"disco lleno {PII[0]}")

    def correr(sink):
        orq, _ = _orq(retriever, _ll("responder", {"respuesta": "Hola, ¿en qué te ayudo?"}), sink=sink)
        r = orq.procesar("hola")
        return r.model_copy(update={"trace_id": ""})

    with caplog.at_level("WARNING"):
        con_fallo = correr(Roto())
    assert con_fallo == correr(None)
    assert "OSError" in caplog.text and PII[0] not in caplog.text


def test_sin_sink_no_escribe_nada(retriever, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TIENDAHOGAR_LOGS_DIR", str(tmp_path / "logs"))
    orq, _ = _orq(retriever, _ll("responder", {"respuesta": "Hola, ¿en qué te ayudo?"}))
    orq.procesar("hola")
    assert not (tmp_path / "logs").exists()
    assert list(tmp_path.iterdir()) == []


def test_logs_sigue_ignorado_por_git():
    lineas = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "logs/" in [x.strip() for x in lineas]


def test_concurrencia_lineas_integras(retriever, tmp_path):
    sink = JsonlTraceSink(tmp_path)
    respuestas = [_ll("responder", {"respuesta": "Hola, ¿en qué te ayudo?"}) for _ in range(40)]
    orq, _ = _orq(retriever, *respuestas, sink=sink)

    def trabajo():
        for _ in range(5):
            orq.procesar("hola")

    hilos = [threading.Thread(target=trabajo) for _ in range(8)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    lineas = (tmp_path / "trazas.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 40
    ids = {json.loads(x)["trace_id"] for x in lineas}
    assert len(ids) == 40


def test_error_inesperado_tambien_traza(retriever):
    sink = FakeTraceSink()
    orq, _ = _orq(retriever, ValueError("explotó"), sink=sink)
    r = orq.procesar("hola")
    assert r.accion == "escalar"
    assert len(sink.trazas) == 1
    assert sink.trazas[0]["accion"] == "escalar" and sink.trazas[0]["trace_id"] == r.trace_id


def test_entrada_vacia_tambien_traza(retriever):
    sink = FakeTraceSink()
    orq, _ = _orq(retriever, sink=sink)
    orq.procesar("   ")
    assert sink.trazas[0]["accion"] == "pedir_dato" and sink.trazas[0]["tokens_entrada"] == 0


def test_settings_rechaza_precios_negativos():
    with pytest.raises(ValidationError):
        Settings(precio_entrada_por_millon=-1)
    with pytest.raises(ValidationError):
        Settings(precio_salida_por_millon=-0.1)


def test_settings_yaml_trae_los_precios():
    assert Settings().precio_entrada_por_millon == 1.0
    assert Settings().precio_salida_por_millon == 5.0
