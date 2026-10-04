"""ADR-008 (acotado): la acción final es coherente con el texto (citar doc5 = escalar; el correo
sin doc5 es `canal_innecesario`: un reintento y, si persiste, escalar).

Todo con FakeLLM, sin red ni API key.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.cli import construir_orquestador
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM, FakeTraceSink
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.guardrail_output import RESPUESTA_SEGURA
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.orquestador import Orquestador, pregunta_por_canal
from tiendahogar_agent.retriever import Retriever

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
CANAL = "soporte@tiendahogar.example"
_contador = itertools.count(1)


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("USAR_CLASIFICADOR_LLM", "MAX_ITERACIONES_LLM", "LLM_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TIENDAHOGAR_LOGS_DIR", str(tmp_path / "logs"))  # T17: sin trazas en el repo


@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


def _responder(texto, fuentes=None, **extra):
    args = {"respuesta": texto, **extra}
    if fuentes is not None:
        args["fuentes"] = fuentes
    return FakeLLM.llamada_tool("responder", args, id=f"c{next(_contador)}")


def _procesar(retriever, mensaje, *respuestas):
    llm = FakeLLM(list(respuestas))
    orq = Orquestador(llm, retriever, Settings(usar_clasificador_llm=False))
    return orq.procesar(mensaje), llm


# ---------------------------------------------------------------- remisión en el texto
MSG_GOLPE = "Mi microondas llegó con un golpe, ¿qué hago?"
LIMPIO = "Lamento lo del golpe; cuéntame cómo llegó el empaque y vemos qué sigue."


def test_correo_sin_doc5_se_reintenta_y_si_el_llm_lo_quita_responde(retriever):
    r, llm = _procesar(
        retriever, MSG_GOLPE,
        _responder(f"Para eso, escríbele a {CANAL} y te ayudarán.", []),
        _responder(LIMPIO, []),
    )
    assert r.accion == "responder" and r.canal is None and CANAL not in r.respuesta
    assert len(llm.llamadas) == 2  # exactamente una llamada extra
    assert "canal_innecesario" in str(llm.llamadas[1]["mensajes"][-1])


def test_reintento_se_detecta_sin_tildes_ni_mayusculas(retriever):
    r, llm = _procesar(
        retriever, MSG_GOLPE,
        _responder("Escribe a SOPORTE@TiendaHogar.Example por favor.", []),
        _responder(LIMPIO, []),
    )
    assert r.accion == "responder" and len(llm.llamadas) == 2


def test_reintento_persiste_escala_con_el_texto_del_llm(retriever):
    texto = f"Para eso, escríbele a {CANAL} y te ayudarán."
    r, llm = _procesar(
        retriever, MSG_GOLPE, _responder(texto, []), _responder(texto, []),
    )
    assert len(llm.llamadas) == 2  # nunca más de una llamada extra
    assert r.accion == "escalar" and r.canal == CANAL
    assert r.respuesta == texto and r.respuesta != RESPUESTA_SEGURA


def test_reintento_persiste_pero_otras_reglas_siguen_aplicando(retriever):
    texto = f"Te lo devolvemos en 99 días, escríbele a {CANAL}."
    r, _ = _procesar(retriever, MSG_GOLPE, _responder(texto, []), _responder(texto, []))
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA


def test_cita_doc5_escala_sin_reintento(retriever):
    r, llm = _procesar(
        retriever, "Tengo un tema de facturación raro con mi compra",
        _responder(f"Escríbele a {CANAL}.", ["doc5"]),
    )
    assert r.accion == "escalar" and r.canal == CANAL and len(llm.llamadas) == 1


def test_cita_doc5_sin_email_en_texto_escala_con_plantilla_segura(retriever):
    """Comportamiento definido: acción escalar, y como el texto no trae el canal R_CANAL lo reemplaza."""
    msg = "Tengo un tema de facturación raro con mi compra"
    r, llm = _procesar(retriever, msg, _responder("Eso lo ve un agente humano.", ["doc5"]))
    assert r.accion == "escalar" and r.canal == CANAL and len(llm.llamadas) == 1
    assert r.respuesta == RESPUESTA_SEGURA and CANAL in r.respuesta


def test_sugerida_escalar_con_correo_no_dispara_reintento(retriever):
    r, llm = _procesar(
        retriever, MSG_GOLPE,
        _responder(f"Escríbele a {CANAL}.", [], accion_sugerida="escalar"),
    )
    assert r.accion == "escalar" and r.canal == CANAL and len(llm.llamadas) == 1


def test_el_reintento_queda_en_la_traza(retriever):
    sink = FakeTraceSink()
    llm = FakeLLM([
        _responder(f"Escríbele a {CANAL}.", []),
        _responder(LIMPIO, []),
    ])
    orq = Orquestador(llm, retriever, Settings(usar_clasificador_llm=False), trace_sink=sink)
    orq.procesar(MSG_GOLPE)
    traza = sink.trazas[-1]
    assert traza["reintentos"] == 1
    assert "canal_innecesario" in traza["reglas_fallidas"]


def test_hecho_y_canal_fallan_juntos_se_atiende_primero_hecho(retriever):
    r, llm = _procesar(
        retriever, "¿Cuánto dura la garantía de una licuadora?",
        _responder(f"Tu licuadora tiene 12 meses de garantía; escríbele a {CANAL}.", ["doc1"]),
        _responder(f"Tu licuadora tiene 6 meses de garantía; escríbele a {CANAL}.", ["doc1"]),
        _responder("Tu licuadora tiene 6 meses de garantía.", ["doc1"]),
    )
    assert len(llm.llamadas) == 3 and r.accion == "responder" and CANAL not in r.respuesta


# ---------------------------------------------------------------- excepción: pregunta de canal
def test_pregunta_de_canal_con_doc5_conserva_responder(retriever):
    texto = f"Puedes escribirle a {CANAL}."
    r, _ = _procesar(
        retriever, "¿Cuál es el canal de contacto?", _responder(texto, ["doc5"]),
    )
    assert r.accion == "responder" and r.canal is None
    assert r.fuentes == ["doc5"] and CANAL in r.respuesta


def test_pregunta_de_canal_sin_cita_y_con_email_conserva_responder(retriever):
    r, _ = _procesar(
        retriever, "¿A qué correo de soporte escribo?", _responder(f"Escribe a {CANAL}.", []),
    )
    assert r.accion == "responder" and r.canal is None


@pytest.mark.parametrize(
    ("mensaje", "esperado"),
    [
        ("¿Cuál es el canal de contacto?", True),
        ("¿Qué canales de atención tienen?", True),
        ("¿Cómo puedo contactarlos?", True),
        ("¿Cuál es su teléfono?", True),
        ("¿Cuál es el horario de atención?", True),
        ("¿Dónde escribo para reportar algo?", True),
        ("¿Cómo hablo con soporte?", True),
        ("¿A qué correo de soporte escribo?", True),
        ("Mi microondas llegó con un golpe, ¿qué hago?", False),
        ("Tengo un tema de facturación raro con mi compra", False),
        ("¿Cuánto dura la garantía?", False),
        ("", False),
    ],
)
def test_pregunta_por_canal_heuristica(mensaje, esperado):
    assert pregunta_por_canal(mensaje) is esperado


# ---------------------------------------------------------------- nunca baja
def test_accion_sugerida_escalar_se_mantiene(retriever):
    r, _ = _procesar(
        retriever, "Mi microondas llegó con un golpe, ¿qué hago?",
        _responder(f"Escríbele a {CANAL}.", [], accion_sugerida="escalar"),
    )
    assert r.accion == "escalar" and r.canal == CANAL


def test_escalar_sugerida_no_baja_aunque_sea_pregunta_de_canal(retriever):
    r, _ = _procesar(
        retriever, "¿Cuál es el canal de contacto?",
        _responder(f"Escríbele a {CANAL}.", ["doc5"], accion_sugerida="escalar"),
    )
    assert r.accion == "escalar" and r.canal == CANAL


def test_respuesta_normal_sin_remision_sigue_en_responder(retriever):
    r, _ = _procesar(
        retriever, "¿Cuánto dura la garantía de una lavadora?",
        _responder("Los electrodomésticos grandes tienen 12 meses de garantía.", ["doc1"]),
    )
    assert r.accion == "responder" and r.canal is None


# ---------------------------------------------------------------- reglas intactas
@pytest.mark.parametrize(
    "mensaje",
    [
        "Quiero reportar el trato de un empleado",
        "Me atendió pésimo el vendedor y quiero quejarme",
        "Quiero un reembolso de $800",
        "Quiero hablar con mi abogado",
    ],
)
def test_escalamientos_por_reglas_siguen_igual(retriever, mensaje):
    llm = FakeLLM([FakeLLM.texto(f"Lo revisa nuestro equipo: escríbele a {CANAL}.")])
    r = Orquestador(llm, retriever, Settings(usar_clasificador_llm=False)).procesar(mensaje)
    assert r.accion == "escalar" and r.canal == CANAL


# ---------------------------------------------------------------- clasificador con proveedor fake
def test_clasificador_activado_por_defecto():
    assert Settings().usar_clasificador_llm is True


def test_fabrica_con_fake_apaga_el_clasificador():
    orq = construir_orquestador(Settings(llm_provider="fake"))
    assert orq._settings.usar_clasificador_llm is False
    r = orq.procesar("¿Cuánto dura la garantía de una lavadora?")
    assert r.accion in ("responder", "escalar")
