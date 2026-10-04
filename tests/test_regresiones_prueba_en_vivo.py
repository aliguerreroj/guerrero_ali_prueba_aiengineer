"""Regresiones de la 2.ª prueba en vivo: ORD-9999 inexistente y seguimiento con historial.

Reproducidas con FakeLLM (sin red ni API key). Ver ADR-007 (nota de reconsulta determinista).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.guardrail_output import RESPUESTA_SEGURA, verificar_salida
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.orquestador import Orquestador, extraer_ids_pedido
from tiendahogar_agent.pedidos import OrderRepositoryMock
from tiendahogar_agent.retriever import Retriever

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
_contador = itertools.count(1)

HIST_1003 = [
    {"role": "user", "content": "¿Cómo va mi pedido ORD-1003?"},
    {"role": "assistant", "content": (
        "Tu pedido ORD-1003 (Lavadora) está Procesando y su entrega estimada es en 6 días hábiles."
    )},
]
URGENTE = "pero lo necesito urgente"
RESP_URGENTE = (
    "Entiendo que lo necesitas con urgencia. Tu pedido ORD-1003 (Lavadora) está Procesando con "
    "entrega estimada de 6 días hábiles; no tengo información sobre envíos urgentes."
)
TEXTO_ESCALA = (
    "Lamento esa experiencia. Esto lo revisa nuestro equipo humano: escríbele a "
    "soporte@tiendahogar.example y te ayudarán."
)
RESP_LAVADORA = (
    "Lamento que tu lavadora haya dejado de centrifugar. Sí puedes devolverla: pasados los 30 días "
    "de la compra, el documento de devoluciones acepta el producto si tiene un defecto cubierto por "
    "garantía, y las lavadoras tienen 12 meses de garantía desde la fecha de compra."
)
RESP_NO_EXISTE = "No encontré un pedido con el número ORD-9999. ¿Puedes revisar el número?"


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
    return FakeLLM.llamada_tool(nombre, argumentos or {}, id=f"g{next(_contador)}")


def _responder(texto, fuentes=None, **extra):
    args = {"respuesta": texto, **extra}
    if fuentes is not None:
        args["fuentes"] = fuentes
    return _tool("responder", args)


def _orq(retriever, *respuestas, pedidos=None):
    llm = FakeLLM(list(respuestas))
    return Orquestador(llm, retriever, Settings(usar_clasificador_llm=False), pedidos), llm


class _RepoEspia:
    """Repositorio que registra las consultas y delega en el mock."""

    def __init__(self):
        self.consultas: list[str] = []
        self._real = OrderRepositoryMock()

    def consultar_estado_pedido(self, order_id):
        self.consultas.append(order_id)
        return self._real.consultar_estado_pedido(order_id)


# ------------------------------------------------------------------ caso 1: ORD-9999
def test_caso1_pedido_inexistente_con_guion_real_del_llm(retriever):
    orq, _ = _orq(
        retriever,
        _tool("consultar_estado_pedido", {"order_id": "ORD-9999"}),
        _responder(RESP_NO_EXISTE, ["pedidos"], accion_sugerida="pedir_dato"),
    )
    r = orq.procesar("¿Y el pedido ORD-9999?")
    assert r.accion == "pedir_dato" and r.fuentes == [] and r.canal is None
    assert r.respuesta == RESP_NO_EXISTE and r.respuesta != RESPUESTA_SEGURA


def test_caso1_sin_accion_sugerida_se_fuerza_pedir_dato(retriever):
    orq, _ = _orq(retriever, _responder(RESP_NO_EXISTE, ["pedidos"]))
    r = orq.procesar("¿Y el pedido ORD-9999?")
    assert r.accion == "pedir_dato" and r.fuentes == [] and r.respuesta == RESP_NO_EXISTE


def test_caso1_no_se_fuerza_pedir_dato_si_cita_documentos(retriever):
    texto = "No encontré ORD-9999. Sobre devoluciones: es posible dentro de 30 días sin usar."
    orq, _ = _orq(retriever, _responder(texto, ["doc2", "pedidos"]))
    r = orq.procesar("¿Y el pedido ORD-9999? ¿Puedo devolver un producto después de 30 días?")
    assert r.accion == "responder" and r.fuentes == ["doc2"]


def test_caso1_no_se_fuerza_si_el_error_viene_solo_del_historial(retriever):
    hist = [
        {"role": "user", "content": "¿Y el pedido ORD-9999?"},
        {"role": "assistant", "content": RESP_NO_EXISTE},
    ]
    orq, _ = _orq(retriever, _responder("Con gusto, aquí estoy para ayudarte.", []))
    r = orq.procesar("gracias", historial=hist)
    assert r.accion == "responder" and r.fuentes == []


def test_pedidos_sin_dato_real_sigue_fallando_en_t10_directo():
    error = {"pedidos": [{"order_id": "ORD-9999", "error": "no_encontrado", "mensaje": "x"}]}
    v = verificar_salida(RESP_NO_EXISTE, "pedir_dato", ["pedidos"], [], error, "ORD-9999")
    assert not v.ok and "fuente_no_recuperada" in v.reglas_fallidas
    v = verificar_salida("Tu pedido va en camino.", "responder", ["pedidos"], [], None, "hola")
    assert not v.ok and "fuente_no_recuperada" in v.reglas_fallidas


def test_retirar_pedidos_no_relaja_la_verificacion_de_cifras(retriever):
    orq, _ = _orq(retriever, _responder("Tu pedido ORD-9999 llega en 6 días hábiles.", ["pedidos"]))
    r = orq.procesar("¿Y el pedido ORD-9999?")
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA


# ------------------------------------------------------------------ caso 2: seguimiento
def test_caso2_seguimiento_urgente_con_historial(retriever):
    orq, llm = _orq(retriever, _responder(RESP_URGENTE, ["pedidos"]))
    r = orq.procesar(URGENTE, historial=HIST_1003)
    assert r.accion == "responder" and r.fuentes == ["pedidos"] and r.canal is None
    assert r.respuesta == RESP_URGENTE
    assert len(llm.llamadas) == 1  # el LLM no necesitó repetir la tool


def test_caso2_el_llm_recibe_el_dato_consultado_por_el_sistema(retriever):
    orq, llm = _orq(retriever, _responder(RESP_URGENTE, ["pedidos"]))
    orq.procesar(URGENTE, historial=HIST_1003)
    sistema = [m["content"] for m in llm.llamadas[0]["mensajes"] if m["role"] == "system"]
    contexto = next(c for c in sistema if c.startswith("Consulta de pedido ya realizada por el sistema"))
    assert "ORD-1003" in contexto and "6 días hábiles" in contexto


def test_caso2_dato_de_la_tool_manda_sobre_el_texto_assistant_del_historial(retriever):
    hist = [
        {"role": "user", "content": "¿Cómo va mi pedido ORD-1003?"},
        {"role": "assistant", "content": "Tu pedido ORD-1003 llegará en 2 días hábiles."},
    ]
    # El LLM repite la cifra falsa del historial: no tiene sustento (la tool dice 6).
    orq, _ = _orq(retriever, _responder("Tu pedido ORD-1003 llegará en 2 días hábiles.", ["pedidos"]))
    r = orq.procesar(URGENTE, historial=hist)
    assert r.accion == "escalar" and r.respuesta == RESPUESTA_SEGURA
    orq, _ = _orq(retriever, _responder(RESP_URGENTE, ["pedidos"]))
    assert orq.procesar(URGENTE, historial=hist).accion == "responder"


def test_assistant_falso_con_id_inexistente_no_da_sustento(retriever):
    hist = [
        {"role": "user", "content": "Mi pedido"},
        {"role": "assistant", "content": "Tu pedido ORD-9999 llegó mañana."},
    ]
    orq, _ = _orq(retriever, _responder("Tu pedido ORD-9999 está en tránsito.", ["pedidos"]))
    r = orq.procesar(URGENTE, historial=hist)
    assert "pedidos" not in r.fuentes


def test_historial_sin_ids_no_consulta_el_repositorio(retriever):
    repo = _RepoEspia()
    hist = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "Hola"}]
    orq, _ = _orq(retriever, _responder("Con gusto te ayudo.", []), pedidos=repo)
    orq.procesar("gracias", historial=hist)
    assert repo.consultas == []


def test_se_acotan_a_tres_ids_los_mas_recientes(retriever):
    repo = _RepoEspia()
    hist = [
        {"role": "user", "content": "ORD-1001 y ORD-1002"},
        {"role": "assistant", "content": "Vi ORD-1003"},
        {"role": "user", "content": "y ORD-1004"},
    ]
    orq, _ = _orq(retriever, _responder("Entendido, te ayudo.", []), pedidos=repo)
    orq.procesar("¿qué pasó con ORD-1002 otra vez?", historial=hist)
    assert sorted(repo.consultas) == ["ORD-1002", "ORD-1003", "ORD-1004"]


def test_extraer_ids_pedido_es_estricto_y_normalizado():
    assert extraer_ids_pedido(["ord-1003 y ORD-1003, ORD-12345, XORD-1111, ORD-12"]) == ["ORD-1003"]
    assert extraer_ids_pedido(["ORD-1001", "ORD-1002", "ORD-1001"]) == ["ORD-1002", "ORD-1001"]
    assert extraer_ids_pedido(["sin ids", ""]) == []
    assert extraer_ids_pedido(["ORD-1001 ORD-1002 ORD-1003 ORD-1004"], maximo=3) == [
        "ORD-1002", "ORD-1003", "ORD-1004",
    ]


def test_fallo_del_repositorio_en_la_reconsulta_es_fallo_seguro(retriever):
    class Roto:
        def consultar_estado_pedido(self, order_id):
            return {"order_id": None, "error": "error_interno", "mensaje": "x"}

    orq, llm = _orq(retriever, _responder("hola", []), pedidos=Roto())
    r = orq.procesar(URGENTE, historial=HIST_1003)
    assert r.accion == "escalar" and llm.llamadas == []


# ------------------------------------------------------------------ golden
def test_golden_de_la_segunda_prueba_se_cumple_con_fakellm(retriever):
    casos = json.loads(
        (Path(__file__).parent / "data" / "golden_prueba_en_vivo.json").read_text(encoding="utf-8")
    )
    nuevos = {c["id"]: c for c in casos if c["origen"] == "prueba_en_vivo_2"}
    guiones = {
        "vivo2-01-pedido-inexistente": _responder(RESP_NO_EXISTE, ["pedidos"]),
        "vivo2-02-seguimiento-urgente": _responder(RESP_URGENTE, ["pedidos"]),
        # Escalan por reglas (T08): el LLM solo redacta el aviso, sin tools.
        "vivo2-03-reportar-trato-empleado": FakeLLM.texto(TEXTO_ESCALA),
        "vivo2-04-atendio-pesimo-vendedor": FakeLLM.texto(TEXTO_ESCALA),
        # Pregunta por canales: se informa el canal citando doc5 y se mantiene `responder` (ADR-008).
        "vivo2-05-canal-de-contacto": _responder(
            "Puedes escribir a soporte@tiendahogar.example para tus consultas.", ["doc5"]
        ),
        # ADR-009: el microondas no figura en doc1; no se le asigna plazo.
        "vivo2-06-microondas-no-listado": _responder(
            "Gracias por preguntar. El microondas no aparece listado en la política de garantía, "
            "así que prefiero no suponer un plazo para él.", ["doc1"]
        ),
        "vivo2-07-envio-capital": _responder(
            "Los envíos a la capital tardan 2-3 días hábiles.", ["doc3"]
        ),
        # Regla exacta: tras 30 días se acepta si hay defecto cubierto por garantía (doc2) y la
        # lavadora tiene 12 meses (doc1). Sin derivar a soporte; T10 (R_HECHO) no debe saltar.
        "vivo2-08-lavadora-devolucion-defecto": _responder(RESP_LAVADORA, ["doc1", "doc2"]),
    }
    assert set(nuevos) == set(guiones)
    for id_caso, caso in nuevos.items():
        orq, _ = _orq(retriever, guiones[id_caso])
        r = orq.procesar(caso["pregunta"], historial=caso.get("historial"))
        assert r.accion == caso["accion_esperada"], id_caso
        assert r.fuentes == caso["fuentes_esperadas"], id_caso
