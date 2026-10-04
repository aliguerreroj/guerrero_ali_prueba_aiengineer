"""Pruebas de la tabla de hechos y de la regla `hecho_incorrecto` (ADR-009). Sin red ni API key."""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path

import pytest

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.guardrail_output import (
    R_HECHO,
    RESPUESTA_SEGURA,
    verificar_salida,
)
from tiendahogar_agent.hechos import HECHOS, NO_LISTADOS, detectar_discrepancias
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.mensajes import validar_historial
from tiendahogar_agent.models import Chunk
from tiendahogar_agent.orquestador import Orquestador
from tiendahogar_agent.retriever import Retriever

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "data" / "docs"
EVAL = Path(__file__).parent / "data" / "hechos_eval.json"
_contador = itertools.count(1)

CHUNKS = [
    Chunk(texto="Garantía de 12 meses (grandes) y de 6 meses (pequeños).", doc_id="doc1"),
    Chunk(texto="Envíos a la capital: 2-3 días hábiles. Otras ciudades: 5-7 días hábiles.", doc_id="doc3"),
]


def _ver(texto, fuentes=("doc1",), accion="responder", mensaje=""):
    return verificar_salida(texto, accion, list(fuentes), CHUNKS, None, mensaje)


def _bloquea(texto, **kw):
    r = _ver(texto, **kw)
    return (not r.ok) and R_HECHO in r.reglas_fallidas


# ----------------------------------------------------------------- la tabla
def _norm(texto: str) -> str:
    return texto.replace("\r\n", "\n").rstrip(" \t")


def test_frase_origen_existe_literal_en_los_documentos():
    for h in HECHOS:
        archivo = next(DOCS.glob(f"{h.doc_id}_*.md"))
        contenido = "\n".join(_norm(linea) for linea in archivo.read_text(encoding="utf-8").splitlines())
        assert h.frase_origen in contenido, h.clave


def test_tabla_sin_duplicados_y_cobertura():
    claves = [h.clave for h in HECHOS]
    assert len(claves) == len(set(claves))
    terminos = [t for h in HECHOS for t in h.terminos]
    assert len(terminos) == len(set(terminos))
    assert not set(terminos) & set(NO_LISTADOS)
    productos = {h.clave for h in HECHOS if h.tipo == "garantia"}
    assert productos == {"refrigeradora", "lavadora", "estufa", "licuadora", "plancha", "tostadora"}
    por = {h.clave: h for h in HECHOS}
    assert por["licuadora"].descripcion() == "licuadora: 6 meses (doc1)"
    assert por["envío a la capital"].valor_texto == "2-3 días hábiles"
    assert por["envío a otras ciudades"].valor_texto == "5-7 días hábiles"


def test_terminos_normalizados():
    for h in HECHOS:
        for t in h.terminos:
            assert t == t.lower() and all(ord(c) < 128 for c in t), t


# ------------------------------------------------------- casos del enunciado
def test_licuadora_12_meses_bloqueada_con_hecho_correcto_en_detalles():
    r = _ver("La licuadora tiene 12 meses de garantía.")
    assert not r.ok and R_HECHO in r.reglas_fallidas and r.respuesta == RESPUESTA_SEGURA
    det = " ".join(r.detalles)
    assert "licuadora: 6 meses (doc1)" in det
    assert "Electrodomésticos pequeños (licuadoras, planchas, tostadoras) tienen garantía de 6 meses." in det


def test_refrigeradora_12_meses_pasa():
    assert _ver("La refrigeradora tiene 12 meses de garantía.").ok


@pytest.mark.parametrize("texto", [
    "Los envíos a la capital tardan 5-7 días.",
    "Envíos a la capital en 5-7 días.",
    "La lavadora tiene 6 meses de garantía.",
    "La plancha tiene 12 meses de garantía.",
    "La tostadora tiene doce meses de garantía.",
    "La estufa tiene 6 meses de garantía.",
    "El microondas tiene garantía de 12 meses.",
    "Las licuadoras tienen 12 meses de garantía.",
])
def test_respuestas_incorrectas_bloqueadas(texto):
    assert _bloquea(texto, fuentes=("doc1", "doc3"))


@pytest.mark.parametrize("texto", [
    "Los envíos a otras ciudades tardan 5-7 días hábiles.",
    "Los envíos a la capital llegan en 2-3 días hábiles.",
    "Los envíos a la capital llegan en 2 a 3 días hábiles.",
    "Los envíos a la capital llegan entre 2 y 3 días hábiles.",
    "Los envíos a la capital llegan en 3 días hábiles.",
    "El microondas no aparece en la política de garantía.",
    "La lavadora tiene 12 meses de garantía.",
    "La licuadora tiene seis meses de garantía.",
])
def test_respuestas_correctas_pasan(texto):
    assert _ver(texto, fuentes=("doc1", "doc3")).ok


def test_destino_otras_ciudades_con_rango_capital_falla():
    assert _bloquea("Los envíos a otras ciudades llegan en 2-3 días hábiles.", fuentes=("doc3",))
    assert _bloquea("Los envíos fuera de la capital llegan en 2-3 días hábiles.", fuentes=("doc3",))


def test_dos_productos_asocia_cada_cifra_al_termino_precedente():
    assert _ver("La licuadora tiene 6 meses y la refrigeradora 12 meses.").ok
    assert _ver("La refrigeradora tiene 12 meses y la licuadora 6 meses.").ok
    assert _bloquea("La licuadora tiene 12 meses y la refrigeradora 12 meses.")
    assert _bloquea("La licuadora tiene 6 meses y la refrigeradora 6 meses.")


def test_cifra_sin_termino_precedente_usa_el_siguiente():
    assert _bloquea("En 12 meses vence la garantía de la licuadora.")
    assert _ver("En 6 meses vence la garantía de la licuadora.").ok


def test_negacion_y_correccion():
    assert _ver("La licuadora no tiene 12 meses, sino 6 meses.").ok
    assert _ver("No son 12 meses, son 6 meses para tu licuadora.").ok
    assert _ver("Para las licuadoras la garantía es de 6 meses, no de 12 meses.").ok
    # la negación no debe regalar el error en otra cláusula
    assert _bloquea("No te preocupes, la licuadora tiene 12 meses.")
    assert _bloquea("La licuadora no tiene 6 meses sino 12 meses.")


def test_cifras_de_otra_unidad_o_sin_unidad_no_activan_garantia():
    assert not detectar_discrepancias("La licuadora tiene 12 cuotas.")
    assert not detectar_discrepancias("La licuadora llega en 30 días.")  # sin destino
    assert not detectar_discrepancias("Las licuadoras vienen en 12 colores.")
    assert not detectar_discrepancias("Tu licuadora cuesta 120 pesos y tiene 12 piezas.")
    assert not detectar_discrepancias("La licuadora tiene garantía.")
    assert not detectar_discrepancias("Pagarás la capital en 12 cuotas, tarda 30 meses?")  # sin envío/producto
    # días no comparan contra garantía y meses no contra envíos
    assert not detectar_discrepancias("Los envíos a la capital cubren 12 meses de promoción.")
    assert not detectar_discrepancias("La refrigeradora llega en 30 días porque está agotada.")


def test_dias_de_devolucion_en_oracion_sin_envio_no_activan_envio():
    assert not detectar_discrepancias("Tienes 30 días para consultar tu pedido en la capital.")


def test_hace_n_meses_no_cuenta():
    assert not detectar_discrepancias("Compraste tu licuadora hace 12 meses.")


def test_unidad_con_un_mes_y_decena_en_letras():
    d = detectar_discrepancias("La licuadora tiene treinta y seis meses.")
    assert d and d[0].cifra.startswith("treinta y seis")
    assert detectar_discrepancias("La licuadora tiene un mes de garantía.")


def test_producto_no_listado_con_garantia_afirmada_falla_y_no_listado_sin_cifra_pasa():
    r = _ver("El televisor tiene 12 meses de garantía.")
    assert not r.ok and R_HECHO in r.reglas_fallidas
    assert "no figura en la política de garantía" in " ".join(r.detalles)
    assert _ver("El microondas no aparece en la política de garantía.").ok
    assert _ver("No tengo el dato del televisor, pero las lavadoras tienen 12 meses.").ok


def test_el_mensaje_del_usuario_no_cuenta_solo_la_respuesta():
    r = _ver("Gracias por preguntar, lo reviso con calma.", mensaje="La licuadora tiene 12 meses?")
    assert r.ok


def test_variantes_de_terminos_y_mayusculas_tildes():
    assert _bloquea("LA REFRIGERADORA TIENE 6 MESES DE GARANTÍA.")
    assert _bloquea("Los refrigeradores tienen 6 meses de garantía.")
    assert _bloquea("Tostadoras: 12 meses de garantía.")
    assert _bloquea("Para otra ciudad el envío tarda 2-3 días hábiles.", fuentes=("doc3",))


def test_oraciones_separadas_no_se_mezclan():
    assert _ver("La licuadora tiene 6 meses. La refrigeradora tiene 12 meses.").ok
    assert _ver("La licuadora tiene 6 meses;\nla refrigeradora tiene 12 meses.").ok


def test_no_lanza_con_entradas_raras():
    assert detectar_discrepancias(None) == []  # type: ignore[arg-type]
    assert detectar_discrepancias("") == []
    r = verificar_salida("La licuadora tiene 12 meses", "responder", ["doc1"], CHUNKS, None, "")
    assert R_HECHO in r.reglas_fallidas


def test_rendimiento_entrada_larga_sin_backtracking():
    relleno = "licuadora " * 1000 + "12 " * 3000 + "meses " * 1000 + "1 " * 3000
    t0 = time.perf_counter()
    detectar_discrepancias(relleno[:19000])
    detectar_discrepancias("la licuadora " + "12 " * 6000 + "x")
    detectar_discrepancias("capital llega " + "5 a " * 4000 + "meses")
    detectar_discrepancias("licuadora tiene 12 meses, " * 700)
    assert time.perf_counter() - t0 < 3.0


# ------------------------------------------------------------- métricas
def test_metricas_del_detector_sobre_el_conjunto_fijo():
    datos = json.loads(EVAL.read_text(encoding="utf-8"))
    assert all(set(d) == {"frase", "esperado", "familia"} for d in datos)
    malas = [d for d in datos if d["esperado"] == "incorrecta"]
    buenas = [d for d in datos if d["esperado"] == "correcta"]
    assert len(malas) >= 20 and len(buenas) >= 20
    assert len({d["frase"] for d in datos}) == len(datos)
    perdidas = [d["frase"] for d in malas if not detectar_discrepancias(d["frase"])]
    falsos = [d["frase"] for d in buenas if detectar_discrepancias(d["frase"])]
    deteccion = 1 - len(perdidas) / len(malas)
    fp = len(falsos) / len(buenas)
    assert deteccion >= 0.90, perdidas
    assert fp <= 0.05, falsos


# ------------------------------------------------------------ orquestador
@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("USAR_CLASIFICADOR_LLM", "MAX_ITERACIONES_LLM", "LLM_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)


def _responder(texto, fuentes=("doc1",)):
    return FakeLLM.llamada_tool(
        "responder", {"respuesta": texto, "fuentes": list(fuentes)}, id=f"r{next(_contador)}"
    )


def _orq(retriever, *respuestas, **kw):
    llm = FakeLLM(list(respuestas))
    return Orquestador(llm, retriever, Settings(usar_clasificador_llm=False, **kw)), llm


PREGUNTA = "¿Cuánto dura la garantía de mi licuadora?"


def test_reintento_corrige_y_entrega_la_segunda(retriever, caplog):
    orq, llm = _orq(
        retriever,
        _responder("Tu licuadora tiene 12 meses de garantía."),
        _responder("Tu licuadora, como electrodoméstico pequeño, tiene 6 meses de garantía."),
    )
    with caplog.at_level(logging.WARNING):
        r = orq.procesar(PREGUNTA)
    assert r.accion == "responder" and "6 meses" in r.respuesta and r.fuentes == ["doc1"]
    assert len(llm.llamadas) == 2
    assert llm.llamadas[1]["tool_choice"] == "any"
    herramientas = [m for m in llm.llamadas[1]["mensajes"] if m["role"] == "tool"]
    assert len(herramientas) == 1 and herramientas[0]["es_error"] is True
    contenido = json.loads(herramientas[0]["content"])
    assert contenido["error"] == "hecho_incorrecto"
    assert any("licuadora: 6 meses (doc1)" in h for h in contenido["hecho_correcto"])
    validar_historial([m for m in llm.llamadas[1]["mensajes"] if m["role"] in ("assistant", "tool")])
    registros = [x.getMessage() for x in caplog.records if x.levelno == logging.WARNING]
    assert any("hecho_incorrecto detectado" in m for m in registros)
    assert any("resuelto" in m for m in registros)
    assert PREGUNTA not in " ".join(registros)


def test_dos_respuestas_incorrectas_dan_respuesta_segura_con_dos_llamadas(retriever, caplog):
    orq, llm = _orq(
        retriever,
        _responder("Tu licuadora tiene 12 meses de garantía."),
        _responder("Sí, son 12 meses para tu licuadora."),
        _responder("Tu licuadora tiene 6 meses."),  # no debe consumirse
    )
    with caplog.at_level(logging.WARNING):
        r = orq.procesar(PREGUNTA)
    assert len(llm.llamadas) == 2
    assert r.respuesta == RESPUESTA_SEGURA and r.accion == "escalar"
    assert r.canal == CANAL_ESCALAMIENTO and r.fuentes == []
    assert any("no resolvio" in x.getMessage() for x in caplog.records)


def test_segundo_intento_con_otra_regla_fallida_tambien_es_segura(retriever):
    orq, llm = _orq(
        retriever,
        _responder("Tu licuadora tiene 12 meses de garantía."),
        _responder("Tu licuadora tiene 6 meses de garantía.", fuentes=("doc9",)),
    )
    r = orq.procesar(PREGUNTA)
    assert len(llm.llamadas) == 2 and r.respuesta == RESPUESTA_SEGURA


def test_el_reintento_no_consume_el_tope_de_iteraciones(retriever):
    orq, llm = _orq(
        retriever,
        _responder("Tu licuadora tiene 12 meses de garantía."),
        _responder("Tu licuadora tiene 6 meses de garantía."),
        max_iteraciones_llm=1,
    )
    r = orq.procesar(PREGUNTA)
    assert r.accion == "responder" and "6 meses" in r.respuesta and len(llm.llamadas) == 2


def test_respuesta_correcta_no_dispara_reintento(retriever):
    orq, llm = _orq(retriever, _responder("Tu licuadora tiene 6 meses de garantía."))
    r = orq.procesar(PREGUNTA)
    assert r.accion == "responder" and len(llm.llamadas) == 1


def test_reintento_microondas_a_no_listado(retriever):
    orq, llm = _orq(
        retriever,
        _responder("El microondas tiene 12 meses de garantía."),
        _responder(
            f"El microondas no aparece listado en la política de garantía, así que prefiero no "
            f"suponer un plazo; escríbenos a {CANAL_ESCALAMIENTO}.",
            fuentes=("doc1",),
        ),
    )
    r = orq.procesar("Me dijeron que el microondas tiene 12 meses de garantía, ¿cierto?")
    assert len(llm.llamadas) == 2 and "no aparece listado" in r.respuesta


# ------------------------------------------------------ grupos coordinados
@pytest.mark.parametrize("texto", [
    "Las licuadoras y las refrigeradoras tienen 12 meses de garantía.",
    "Tanto la licuadora como la lavadora tienen 12 meses de garantía.",
    "La garantía de tu licuadora es la misma que la de la lavadora: 12 meses.",
    "La refrigeradora y la licuadora tienen 12 meses de garantía.",
    "Las licuadoras, planchas y estufas tienen 6 meses de garantía.",
    "En 12 meses vence la garantía de la licuadora y la lavadora.",
    "A la capital y a otras ciudades llega en 5-7 días hábiles.",
])
def test_grupo_coordinado_se_evalua_contra_todos_los_terminos(texto):
    assert _bloquea(texto, fuentes=("doc1", "doc3"))


@pytest.mark.parametrize("texto", [
    "Las lavadoras y las refrigeradoras tienen 12 meses de garantía.",
    "Tanto la lavadora como la estufa tienen 12 meses de garantía.",
    "Las licuadoras, planchas y tostadoras tienen 6 meses de garantía.",
    "La licuadora tiene 6 meses y la refrigeradora 12 meses.",
    "La licuadora tiene 6 meses, la plancha 6 meses y la lavadora 12 meses.",
])
def test_grupo_coherente_o_cifras_separadas_pasan(texto):
    assert _ver(texto).ok


def test_microondas_con_respuesta_honesta_sin_canal_queda_en_responder(retriever):
    orq, llm = _orq(
        retriever,
        _responder(
            "El microondas no aparece listado en la política de garantía, así que no puedo "
            "asignarle un plazo.", fuentes=("doc1",),
        ),
    )
    r = orq.procesar("¿El microondas tiene 12 meses de garantía?")
    assert r.accion == "responder" and r.canal is None and len(llm.llamadas) == 1


# ----------------------------------- dos cifras correctas: no se agrupan (ADR-009)
@pytest.mark.parametrize("texto", [
    "La licuadora y la lavadora tienen 6 y 12 meses respectivamente.",
    "La licuadora y la lavadora tienen 6 meses y 12 meses.",
    "Las licuadoras y las refrigeradoras tienen 6 y 12 meses respectivamente.",
    "Tienes 12 meses de garantía en tu lavadora y 6 meses en tu licuadora.",
    "Tienes 6 meses de garantía en tu licuadora y 12 meses en tu lavadora.",
    "El envío tarda 2-3 días hábiles a la capital y 5-7 días hábiles a otras ciudades.",
    "Llega a otras ciudades en 5-7 días hábiles y a la capital en 2-3 días hábiles.",
    "A la capital y a otras ciudades llega en 2-3 días hábiles y 5-7 días hábiles respectivamente.",
])
def test_dos_cifras_correctas_con_dos_terminos_pasan(texto):
    assert _ver(texto, fuentes=("doc1", "doc3")).ok


@pytest.mark.parametrize("texto", [
    "La licuadora y la lavadora tienen 12 y 6 meses respectivamente.",
    "La licuadora y la lavadora tienen 12 meses y 6 meses.",
    "Tienes 12 meses de garantía en tu licuadora y 6 meses en tu lavadora.",
    "El envío tarda 5-7 días hábiles a la capital y 2-3 días hábiles a otras ciudades.",
    "Las licuadoras y las lavadoras tienen entre 6 y 12 meses.",
])
def test_dos_cifras_cruzadas_siguen_fallando(texto):
    assert _bloquea(texto, fuentes=("doc1", "doc3"))


def test_cifras_y_terminos_en_distinto_numero_no_se_evaluan():
    # documentado: n términos y m cifras con n distinto de m no se emparejan (lado seguro)
    assert not detectar_discrepancias("La licuadora, la plancha y la lavadora tienen 6 y 12 meses.")


# ----------------------------------------- conectores, atribución y coste
@pytest.mark.parametrize("texto", [
    "La licuadora, así como la lavadora, tiene 12 meses de garantía.",
    "La licuadora tiene lo mismo que la lavadora: 12 meses.",
    "La licuadora tiene igual garantía que la lavadora: 12 meses.",
    "A la capital, así como a otras ciudades, llega en 5-7 días hábiles.",
    "La licuadora tiene 12 meses de garantía, como dices.",
])
def test_conectores_asi_como_y_lo_mismo_que(texto):
    assert _bloquea(texto, fuentes=("doc1", "doc3"))


@pytest.mark.parametrize("texto", [
    "Dices que la licuadora tiene 12 meses, pero son 6 meses.",
    "Mencionas una licuadora con 12 meses, pero en realidad su garantía es de 6.",
    "Me dijeron que la plancha tiene 12 meses, pero la garantía es de 6 meses.",
    "Según tú la licuadora tiene 12 meses; en los documentos son 6 meses.",
    "Escuchaste que la licuadora tiene doce meses, pero tiene 6 meses.",
])
def test_cifra_atribuida_al_cliente_no_se_evalua(texto):
    assert _ver(texto).ok


def test_atribucion_no_exime_la_afirmacion_propia():
    assert _bloquea("Dices que sí, y la licuadora tiene 12 meses.")
    assert _bloquea("La licuadora tiene 12 meses, como dices.")


def test_rendimiento_con_grupos_y_cifras_repetidas():
    t0 = time.perf_counter()
    detectar_discrepancias("licuadora, lavadora, " * 5000 + "6 meses " * 5000)
    detectar_discrepancias(("la licuadora tiene 12 meses y la lavadora 6 meses, " * 2000)[:100_000])
    detectar_discrepancias("capital, " * 10000 + "llega en 2-3 días " * 3000)
    detectar_discrepancias("12 meses " * 11000)
    assert time.perf_counter() - t0 < 2.0
