"""ADR-008: la acción final es coherente con el texto (remitir a soporte = escalar).

Todo con FakeLLM, sin red ni API key.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.cli import construir_orquestador
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
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
def test_texto_que_remite_a_soporte_escala_con_canal(retriever):
    texto = f"Para eso, escríbele a {CANAL} y te ayudarán."
    r, _ = _procesar(
        retriever, "Mi microondas llegó con un golpe, ¿qué hago?",
        _responder(texto, [], accion_sugerida="responder"),
    )
    assert r.accion == "escalar" and r.canal == CANAL and CANAL in r.respuesta


def test_remision_se_detecta_sin_tildes_ni_mayusculas(retriever):
    r, _ = _procesar(
        retriever, "Mi microondas llegó con un golpe, ¿qué hago?",
        _responder("Escribe a SOPORTE@TiendaHogar.Example por favor.", []),
    )
    assert r.accion == "escalar" and r.canal == CANAL


def test_cita_doc5_sin_email_en_texto_escala_con_plantilla_segura(retriever):
    """Comportamiento definido: acción escalar, y como el texto no trae el canal R_CANAL lo reemplaza."""
    msg = "Tengo un tema de facturación raro con mi compra"
    r, _ = _procesar(retriever, msg, _responder("Eso lo ve un agente humano.", ["doc5"]))
    assert r.accion == "escalar" and r.canal == CANAL
    assert r.respuesta == RESPUESTA_SEGURA and CANAL in r.respuesta


def test_cita_doc5_con_email_en_texto_escala(retriever):
    msg = "Tengo un tema de facturación raro con mi compra"
    r, _ = _procesar(retriever, msg, _responder(f"Escríbele a {CANAL}.", ["doc5"]))
    assert r.accion == "escalar" and r.canal == CANAL


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
