"""Orquestador: RAG por turno, tool obligatoria, fuera de alcance y fuente «pedidos».

Casos reales de una prueba con un LLM de verdad, reproducidos con FakeLLM (sin red ni API key).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.clasificador import NOMBRE_TOOL as TOOL_CLASIFICAR
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.excepciones import ErrorLLMServidor
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.guardrail_output import RESPUESTA_SEGURA
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import Chunk
from tiendahogar_agent.orquestador import (
    MENSAJE_FUERA_DE_ALCANCE,
    TOOL_CHOICE_BUCLE,
    Orquestador,
    _Evidencia,
)
from tiendahogar_agent.resiliencia import MENSAJE_FALLO_SEGURO
from tiendahogar_agent.retriever import Retriever

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
_contador = itertools.count(1)

LICUADORA = "Me dijeron que la licuadora tiene 12 meses de garantía, ¿cierto?"
MUNDIAL = "¿Quién ganó el mundial?"
CORRECCION = (
    "Gracias por preguntar. Para las licuadoras la garantía es de 6 meses desde la fecha de "
    "compra, no de 12 meses, y cubre defectos de fábrica."
)
AMABLE = (
    "Gracias por escribirnos. Ese tema se sale de lo que puedo resolver, pero con gusto te ayudo "
    "con garantía, devoluciones, envíos, reembolsos, canales de contacto o el estado de un pedido."
)


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("USAR_CLASIFICADOR_LLM", "MAX_ITERACIONES_LLM", "LLM_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


def _tool(nombre, argumentos=None):
    return FakeLLM.llamada_tool(nombre, argumentos or {}, id=f"r{next(_contador)}")


def _responder(texto, fuentes=None, **extra):
    args = {"respuesta": texto, **extra}
    if fuentes is not None:
        args["fuentes"] = fuentes
    return _tool("responder", args)


def _orq(retriever, *respuestas, **kw):
    llm = FakeLLM(list(respuestas))
    kw.setdefault("usar_clasificador_llm", False)  # ADR-008: por defecto activado; apagado para guiar solo el bucle
    return Orquestador(llm, retriever, Settings(**kw)), llm


# ---------------------------------------------------------------- RAG por turno
def test_licuadora_se_corrige_con_doc1_sin_llamar_buscar_politicas(retriever):
    orq, llm = _orq(retriever, _responder(CORRECCION, ["doc1"]))
    r = orq.procesar(LICUADORA)
    assert r.accion == "responder" and r.fuentes == ["doc1"] and r.canal is None
    assert "6 meses" in r.respuesta
    assert len(llm.llamadas) == 1  # el LLM nunca llamó buscar_politicas


def test_contexto_del_turno_se_inyecta_con_doc_id_real(retriever):
    orq, llm = _orq(retriever, _responder(CORRECCION, ["doc1"]))
    orq.procesar(LICUADORA)
    mensajes = llm.llamadas[0]["mensajes"]
    assert [m["role"] for m in mensajes] == ["system", "system", "user"]
    contexto = mensajes[1]["content"]
    assert '<documento id="doc1" fuente="doc1">' in contexto and "6 meses" in contexto
    assert mensajes[2] == {"role": "user", "content": LICUADORA}


def test_el_retriever_corre_en_cada_turno(retriever):
    historial = [
        {"role": "user", "content": "¿Cuánto tarda un envío a la capital?"},
        {"role": "assistant", "content": "Los envíos a la capital tardan 2-3 días hábiles."},
    ]
    orq, llm = _orq(retriever, _responder(CORRECCION, ["doc1"]))
    r = orq.procesar(LICUADORA, historial=historial)
    assert r.accion == "responder" and r.fuentes == ["doc1"]
    contexto = llm.llamadas[0]["mensajes"][1]["content"]
    assert 'id="doc1"' in contexto and 'id="doc3"' not in contexto


def test_cita_de_doc_no_recuperado_en_el_turno_sigue_fallando(retriever):
    # doc4 no se recuperó para este mensaje: T10 lo rechaza aunque el doc exista.
    orq, _ = _orq(retriever, _responder(CORRECCION, ["doc4"]))
    r = orq.procesar(LICUADORA)
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA and r.fuentes == []


def test_cifra_sin_sustento_en_el_turno_sigue_fallando(retriever):
    # ADR-009: la cifra contradice la tabla de hechos, hay un reintento y vuelve a fallar.
    orq, _ = _orq(
        retriever,
        _responder("Las licuadoras tienen 24 meses de garantía.", ["doc1"]),
        _responder("Las licuadoras tienen 24 meses de garantía.", ["doc1"]),
    )
    r = orq.procesar(LICUADORA)
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA


def test_buscar_politicas_suma_evidencia_al_turno(retriever):
    # «hola» no recupera nada; lo que busque el LLM después sí sustenta la respuesta.
    orq, llm = _orq(
        retriever,
        _tool("buscar_politicas", {"consulta": "envíos a otras ciudades"}),
        _responder("A otras ciudades el envío tarda 5-7 días hábiles.", ["doc3"]),
    )
    r = orq.procesar("hola")
    assert r.accion == "responder" and r.fuentes == ["doc3"]
    assert "No se recuperaron documentos relevantes" in llm.llamadas[0]["mensajes"][1]["content"]


def test_evidencia_no_duplica_chunks():
    ev = _Evidencia()
    c = Chunk(texto="uno", doc_id="doc1", metadatos={"posicion": 0})
    ev.agregar_chunks([c])
    ev.agregar_chunks([c.model_copy(), Chunk(texto="dos", doc_id="doc1", metadatos={"posicion": 1})])
    assert [x.texto for x in ev.chunks] == ["uno", "dos"]


def test_chunks_de_otro_turno_no_sustentan(retriever):
    # Un mismo Orquestador atiende dos turnos: la evidencia del primero no pasa al segundo.
    llm = FakeLLM([
        _responder("Los envíos a la capital tardan 2-3 días hábiles.", ["doc3"]),
        _responder("Los envíos a la capital tardan 2-3 días hábiles.", ["doc3"]),
    ])
    orq = Orquestador(llm, retriever, Settings(usar_clasificador_llm=False))
    assert orq.procesar("¿Cuánto tarda un envío a la capital?").accion == "responder"
    r = orq.procesar(LICUADORA)
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA


def test_contexto_vacio_lo_dice_explicitamente(retriever):
    orq, llm = _orq(retriever, _responder(AMABLE))
    orq.procesar(MUNDIAL)
    contexto = llm.llamadas[0]["mensajes"][1]["content"]
    assert "No se recuperaron documentos relevantes" in contexto and "<documento" not in contexto


# ---------------------------------------------------------------- tool_choice
def test_bucle_pide_tool_obligatoria_y_el_clasificador_no(retriever):
    assert TOOL_CHOICE_BUCLE == "any"
    orq, llm = _orq(
        retriever,
        _tool(TOOL_CLASIFICAR, {"intencion": "politica"}),
        _tool("buscar_politicas", {"consulta": "garantía licuadora"}),
        _responder(CORRECCION, ["doc1"]),
        usar_clasificador_llm=True,
    )
    r = orq.procesar(LICUADORA)
    assert r.accion == "responder"
    assert [c["tool_choice"] for c in llm.llamadas] == [None, "any", "any"]


def test_texto_suelto_se_reintenta_con_recordatorio_y_luego_responde(retriever):
    orq, llm = _orq(retriever, FakeLLM.texto("Son 6 meses."), _responder(CORRECCION, ["doc1"]))
    r = orq.procesar(LICUADORA)
    assert r.accion == "responder" and r.fuentes == ["doc1"] and len(llm.llamadas) == 2
    ultimo = llm.llamadas[1]["mensajes"][-1]
    assert ultimo["role"] == "user" and "herramienta `responder`" in ultimo["content"]
    assert all(c["tool_choice"] == "any" for c in llm.llamadas)


def test_tool_obligatoria_no_evita_el_tope_de_iteraciones(retriever):
    orq, llm = _orq(
        retriever, *[_tool("buscar_politicas", {"consulta": f"garantía {i}"}) for i in range(10)],
        max_iteraciones_llm=3,
    )
    r = orq.procesar("¿Cuánto dura la garantía de una lavadora?")
    assert r.accion == "escalar" and r.respuesta == MENSAJE_FALLO_SEGURO
    assert len(llm.llamadas) == 3


# ---------------------------------------------------------------- fuera de alcance
def test_mundial_con_llm_que_responde_amable(retriever):
    orq, llm = _orq(retriever, _responder(AMABLE))
    r = orq.procesar(MUNDIAL)
    assert r.accion == "responder" and r.fuentes == [] and r.canal is None
    assert r.respuesta == AMABLE and "problema técnico" not in r.respuesta
    assert len(llm.llamadas) == 1


@pytest.mark.parametrize(
    "guion",
    [
        [FakeLLM.texto("El mundial lo ganó..."), FakeLLM.texto("No sé de fútbol")],
        [FakeLLM.texto("No tengo ese dato"), FakeLLM.texto("No tengo ese dato")],
    ],
)
def test_mundial_con_texto_suelto_usa_la_plantilla_amable(retriever, guion):
    orq, llm = _orq(retriever, *guion)
    r = orq.procesar(MUNDIAL)
    assert r.accion == "responder" and r.respuesta == MENSAJE_FUERA_DE_ALCANCE
    assert r.fuentes == [] and r.canal is None and len(llm.llamadas) == 2


def test_iteraciones_agotadas_sin_evidencia_es_falla_real(retriever):
    # Bucle de búsquedas vacías hasta agotar iteraciones: el LLM se rompió, no es un tema ajeno.
    orq, llm = _orq(
        retriever, *[_tool("buscar_politicas", {"consulta": "mundial de fútbol"}) for _ in range(10)],
    )
    r = orq.procesar(MUNDIAL)
    assert r.accion == "escalar" and r.respuesta == MENSAJE_FALLO_SEGURO
    assert len(llm.llamadas) == Settings().max_iteraciones_llm


def test_tool_calls_invalidos_hasta_agotar_iteraciones_es_falla_real(retriever):
    # Regresión B1 (2): argumentos inválidos repetidos NO son fuera de alcance.
    orq, _ = _orq(retriever, *[_tool("consultar_estado_pedido", {}) for _ in range(10)])
    r = orq.procesar("blah xyz")
    assert r.accion == "escalar" and r.respuesta == MENSAJE_FALLO_SEGURO


def test_texto_suelto_con_pedido_en_el_mensaje_es_falla_real(retriever):
    # Regresión B1 (1): hay un id de pedido, así que no es un tema ajeno.
    orq, _ = _orq(retriever, FakeLLM.texto("hola"), FakeLLM.texto("hola"))
    r = orq.procesar("¿Dónde está ORD-1001?")
    assert r.accion == "escalar" and r.respuesta == MENSAJE_FALLO_SEGURO


def test_texto_suelto_tras_intentar_una_tool_es_falla_real(retriever):
    orq, _ = _orq(
        retriever, _tool("buscar_politicas", {"consulta": "mundial de fútbol"}),
        FakeLLM.texto("no sé"), FakeLLM.texto("no sé"),
    )
    r = orq.procesar(MUNDIAL)
    assert r.accion == "escalar" and r.respuesta == MENSAJE_FALLO_SEGURO


def test_hola_sin_documentos_ni_tools_recibe_la_plantilla_amable(retriever):
    # Esperado: «hola» no recupera documentos; si el LLM solo da texto suelto (sin tools, sin
    # pedido en el mensaje) no hay falla de sistema ni nada que escalar: se invita a elegir un
    # tema con la plantilla amable (accion=responder), nunca «problema técnico».
    orq, _ = _orq(retriever, FakeLLM.texto("Hola"), FakeLLM.texto("Hola de nuevo"))
    r = orq.procesar("Hola")
    assert r.accion == "responder" and r.respuesta == MENSAJE_FUERA_DE_ALCANCE
    assert r.fuentes == [] and r.canal is None


def test_fallas_reales_siguen_siendo_problema_tecnico(retriever):
    for fallo in (ErrorLLMServidor("500"), FakeLLM.vacia()):
        orq, _ = _orq(retriever, fallo)
        r = orq.procesar(MUNDIAL)
        assert r.accion == "escalar" and r.respuesta == MENSAJE_FALLO_SEGURO
    # dos «responder» en una respuesta es un LLM roto, no un tema ajeno
    doble = FakeLLM.llamada_tool("responder", {"respuesta": "a"}, id="d1")
    doble.llamadas_tools.append(
        FakeLLM.llamada_tool("responder", {"respuesta": "b"}, id="d2").llamadas_tools[0]
    )
    orq, _ = _orq(retriever, doble)
    assert orq.procesar(MUNDIAL).respuesta == MENSAJE_FALLO_SEGURO


def test_pedido_sin_id_sigue_el_camino_pedir_dato(retriever):
    orq, _ = _orq(
        retriever,
        _responder("Con gusto te ayudo. ¿Me compartes tu número de pedido?",
                   accion_sugerida="pedir_dato"),
    )
    r = orq.procesar("¿Dónde está mi pedido?")
    assert r.accion == "pedir_dato" and r.canal is None


def test_pedido_sin_id_con_llm_caido_en_texto_no_dice_problema_tecnico(retriever):
    # texto suelto dos veces, sin documentos ni pedido: plantilla amable que ofrece ayudar con pedidos
    orq, _ = _orq(retriever, FakeLLM.texto("ok"), FakeLLM.texto("ok"))
    r = orq.procesar("¿Dónde está mi pedido?")
    assert r.accion == "responder" and "pedido" in r.respuesta


# ---------------------------------------------------------------- fuente «pedidos»
def test_consulta_de_pedido_cita_pedidos(retriever):
    orq, llm = _orq(
        retriever,
        _tool("consultar_estado_pedido", {"order_id": "ORD-1001"}),
        _responder("Tu pedido ORD-1001 (Refrigeradora) está En tránsito y llega en 3 días hábiles.",
                   ["pedidos"]),
    )
    r = orq.procesar("¿Dónde está mi pedido ORD-1001?")
    assert r.accion == "responder" and r.fuentes == ["pedidos"]
    resultado = next(m for m in llm.llamadas[1]["mensajes"] if m["role"] == "tool")
    assert json.loads(resultado["content"])["estado"] == "En tránsito"


def test_citar_pedidos_sin_haber_consultado_falla(retriever):
    # Ajustado (ronda 2): con un ORD-#### en el mensaje el sistema ya consulta el pedido, así que
    # citar «pedidos» tendría respaldo; el caso «nunca se consultó» se prueba sin id en el mensaje.
    orq, _ = _orq(retriever, _responder("Tu pedido ya va en camino.", ["pedidos"]))
    r = orq.procesar("¿Dónde está mi pedido?")
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA and r.fuentes == []
    assert r.canal == CANAL_ESCALAMIENTO


def test_citar_pedidos_si_la_tool_devolvio_error_se_retira_la_fuente(retriever):
    # Ajustado (ronda 2): antes escalaba por fuente_no_recuperada; ahora la cita sin respaldo se
    # retira y el turno pasa a pedir_dato (ADR-007). T10 directo sigue rechazándola.
    orq, _ = _orq(
        retriever,
        _tool("consultar_estado_pedido", {"order_id": "ORD-9999"}),
        _responder("No encontré el pedido ORD-9999. ¿Puedes revisar el número?", ["pedidos"]),
    )
    r = orq.procesar("Mi pedido ORD-9999 no aparece")
    assert r.accion == "pedir_dato" and r.fuentes == [] and r.respuesta != RESPUESTA_SEGURA


def test_pedido_inexistente_sin_citar_fuentes_no_escala(retriever):
    orq, _ = _orq(
        retriever,
        _tool("consultar_estado_pedido", {"order_id": "ORD-9999"}),
        _responder("No encontré el pedido ORD-9999. ¿Puedes revisar el número?",
                   accion_sugerida="pedir_dato"),
    )
    r = orq.procesar("Mi pedido ORD-9999 no aparece")
    assert r.accion == "pedir_dato" and r.fuentes == []


def test_pedido_y_documento_en_el_mismo_turno(retriever):
    orq, _ = _orq(
        retriever,
        _tool("consultar_estado_pedido", {"order_id": "ORD-1002"}),
        _responder(
            "Tu pedido ORD-1002 (Licuadora) fue entregado; la garantía de licuadoras es de 6 meses.",
            ["doc1", "pedidos"],
        ),
    )
    r = orq.procesar("Mi licuadora del pedido ORD-1002, ¿cuánto dura la garantía?")
    assert r.accion == "responder" and r.fuentes == ["doc1", "pedidos"]
