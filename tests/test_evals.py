"""Pruebas del runner de evals y de la calibración (T19): FakeLLM, sin red ni API key.

El LLM real NUNCA se llama aquí: el runner recibe una fábrica de orquestadores inyectable y los
casos usan los guiones de `golden_guiones.py`. Ningún test lee el `.env` real (cwd temporal y
`Settings(_env_file=None, ...)`).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from golden_guiones import GUIONES, IDS_RETRIEVAL_AMPLIO, construir_retriever_amplio

from tiendahogar_agent import calibracion, evals
from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM, FakeTraceSink
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.evals import SinkMemoria
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.matching import contiene, faltantes, normalizar, presentes
from tiendahogar_agent.models import Chunk
from tiendahogar_agent.orquestador import Orquestador
from tiendahogar_agent.retriever import Retriever
from tiendahogar_agent.texto import quitar_tildes

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "data" / "docs"
GOLDEN = json.loads((RAIZ / "evals" / "golden_set.json").read_text(encoding="utf-8"))
CLAVE_FALSA = "sk-ant-clave-falsa-0123456789"


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("USAR_CLASIFICADOR_LLM", "MAX_ITERACIONES_LLM", "LLM_PROVIDER", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)  # un .env real nunca se lee


@pytest.fixture(scope="module")
def retriever():
    chunks = FileSystemDocumentSource(DOCS).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=3, umbral_bm25=0.5, umbral_semantico=2.0,
    )


def _settings(**kw) -> Settings:
    kw.setdefault("anthropic_api_key", None)
    return Settings(_env_file=None, **kw)


def _fabrica_guiones(retriever):
    """Fábrica con el guion de cada caso (los que escalan por reglas usan un LLM vacío)."""
    def fabrica(caso, sink):
        guion = GUIONES.get(caso["id"])
        if guion is None:
            llm, clasif = FakeLLM([FakeLLM.vacia()]), False
        else:
            llm, clasif = FakeLLM(list(guion.respuestas)), guion.clasificador
        recuperador = construir_retriever_amplio(DOCS) if caso["id"] in IDS_RETRIEVAL_AMPLIO else retriever
        return Orquestador(llm, recuperador, Settings(_env_file=None, usar_clasificador_llm=clasif),
                           trace_sink=sink)
    return fabrica


@pytest.fixture(scope="module")
def reporte_guiones(retriever):
    return evals.ejecutar_evals(GOLDEN, _fabrica_guiones(retriever), _settings())


# ------------------------------------------------------------------ matching
def test_matching_normaliza_tildes_mayusculas_y_espacios():
    assert normalizar("  Número   DE  Pedido\n") == "numero de pedido"
    assert contiene("¿Me compartes el NÚMERO de pedido?", "numero de pedido")


def test_matching_alternativas_con_barra():
    patron = "numero de pedido|numero del pedido"
    assert contiene("Dame el número del pedido", patron)
    assert contiene("Dame el número de pedido", patron)
    assert not contiene("Dame el código", patron)
    # La alternativa vacía no cuenta (una barra suelta no hace que todo coincida).
    assert not contiene("hola", "adios|")


def test_matching_faltantes_y_presentes():
    texto = "Tienes 6 meses de garantía."
    assert faltantes(texto, ["6 meses", "devoluciones|garantia", "12 meses"]) == ["12 meses"]
    assert presentes(texto, ["12 meses", "garantia|devolucion"]) == ["garantia|devolucion"]


# ------------------------------------------------------------------ respaldo en la traza
def _orq(retriever, *respuestas, clasificador=False):
    sink = FakeTraceSink()
    llm = FakeLLM(list(respuestas))
    orq = Orquestador(llm, retriever, Settings(_env_file=None, usar_clasificador_llm=clasificador),
                      trace_sink=sink)
    return orq, sink


def _responder(texto, fuentes):
    return FakeLLM.llamada_tool("responder", {"respuesta": texto, "fuentes": fuentes}, id="r1")


def test_respaldo_false_en_respuesta_normal(retriever):
    orq, sink = _orq(retriever, _responder("Las lavadoras tienen 12 meses de garantía.", ["doc1"]))
    r = orq.procesar("¿Cuánto dura la garantía de una lavadora?")
    assert r.accion == "responder"
    assert sink.trazas[0]["respaldo"] is False


def test_respaldo_true_con_plantilla_por_llm_vacio_y_false_si_el_llm_redacta(retriever):
    orq, sink = _orq(retriever, FakeLLM.vacia())
    orq.procesar("Quiero un reembolso de $800 por mi lavadora.")
    assert sink.trazas[0]["respaldo"] is True and sink.trazas[0]["accion"] == "escalar"

    orq, sink = _orq(retriever, FakeLLM.texto(
        "Entiendo, un supervisor humano revisará tu caso: escribe a soporte@tiendahogar.example."))
    r = orq.procesar("Quiero un reembolso de $800 por mi lavadora.")
    assert r.accion == "escalar" and "supervisor" in r.respuesta
    assert sink.trazas[0]["respaldo"] is False  # escalamiento legítimo, no respaldo


def test_respaldo_true_si_falla_la_verificacion_de_salida(retriever):
    cifra_inventada = _responder("Las lavadoras tienen 99 meses de garantía.", ["doc1"])
    orq, sink = _orq(retriever, cifra_inventada)
    r = orq.procesar("¿Cuánto dura la garantía de una lavadora?")
    assert r.accion == "escalar"
    assert sink.trazas[0]["respaldo"] is True and sink.trazas[0]["reglas_fallidas"]


def test_respaldo_true_en_fallo_seguro_por_error_del_llm(retriever):
    orq, sink = _orq(retriever, RuntimeError("caída"))
    r = orq.procesar("¿Cuánto dura la garantía de una lavadora?")
    assert r.accion == "escalar"
    assert sink.trazas[0]["respaldo"] is True


# ------------------------------------------------------------------ métricas con guiones
POR_CATEGORIA = {
    "politica": 10, "pedido": 5, "escalamiento": 10,
    "fuera_de_alcance": 5, "manipulacion": 2, "hecho_critico": 5,
}


def test_golden_tiene_el_tamano_calculado_a_mano():
    assert len(GOLDEN) == 37
    for cat, n in POR_CATEGORIA.items():
        assert sum(c["categoria"] == cat for c in GOLDEN) == n


def test_metricas_por_categoria_con_los_guiones(reporte_guiones):
    rep = reporte_guiones
    assert rep["golden_casos"] == 37 and rep["casos_ejecutados"] == 37 and not rep["abortado"]
    assert set(rep["por_categoria"]) == set(POR_CATEGORIA)
    for cat, n in POR_CATEGORIA.items():
        m = rep["por_categoria"][cat]
        assert m["n"] == n
        # Guiones bien portados: todo cumple, sin dejar nada a la suerte.
        for k in ("accion_acierto", "fuentes_acierto", "debe_contener_cumplimiento",
                  "no_debe_contener_cumplimiento", "caso_ok"):
            assert m[k] == 1.0, (cat, k)
        assert m["costo_total_usd"] == 0.0
    g = rep["global"]
    assert g["n"] == 37 and g["caso_ok"] == 1.0
    # Respaldo: los 12 casos que escalan por reglas usan la plantilla (LLM vacío); ningún otro.
    assert rep["por_categoria"]["escalamiento"]["respaldo_tasa"] == 1.0
    assert rep["por_categoria"]["manipulacion"]["respaldo_tasa"] == 1.0
    for cat in ("politica", "pedido", "fuera_de_alcance", "hecho_critico"):
        assert rep["por_categoria"][cat]["respaldo_tasa"] == 0.0, cat
    assert g["respaldo_tasa"] == pytest.approx(12 / 37, abs=1e-4)
    prom = sum(c["latencia_ms"] for c in rep["casos"]) / 37
    assert g["latencia_promedio_ms"] == pytest.approx(prom, abs=0.01)


def test_cada_caso_del_reporte_trae_los_campos_pedidos(reporte_guiones):
    claves = {
        "id", "categoria", "accion_esperada", "accion_obtenida", "accion_ok", "fuentes_esperadas",
        "fuentes_obtenidas", "fuentes_ok", "debe_ok", "debe_faltantes", "no_debe_ok",
        "no_debe_presentes", "respaldo", "costo_usd", "latencia_ms", "respuesta", "conocido",
    }
    for c in reporte_guiones["casos"]:
        assert claves <= set(c), c["id"]


def test_metricas_a_mano_con_fallos_costo_y_conocido(retriever, monkeypatch):
    """Dos casos sintéticos: uno cumple y otro falla en acción, fuentes, debe y no_debe."""
    monkeypatch.setitem(
        evals.CASOS_CONOCIDOS, "t18-17-hecho-capital-y-devolucion",
        "Falso positivo de ejemplo para probar el marcado de casos conocidos.",
    )
    casos = [
        {"id": "a", "categoria": "politica", "origen": "t18",
         "mensajes": [{"role": "user", "content": "¿Cuánto dura la garantía de una lavadora?"}],
         "accion_esperada": "responder", "fuentes_esperadas": ["doc1"],
         "debe_contener": ["12 meses|un anio"], "no_debe_contener": ["6 meses"], "notas": "x"},
        {"id": "t18-17-hecho-capital-y-devolucion", "categoria": "hecho_critico", "origen": "t18",
         "mensajes": [{"role": "user", "content": "¿Cuánto dura la garantía de una lavadora?"}],
         "accion_esperada": "escalar", "fuentes_esperadas": ["doc3"],
         "debe_contener": ["supervisor"], "no_debe_contener": ["12 meses"], "notas": "x"},
    ]
    llms = {
        "a": [FakeLLM.llamada_tool(
            "responder", {"respuesta": "Tienen 12 meses de garantía.", "fuentes": ["doc1"]},
            id="x1", tokens_entrada=1000, tokens_salida=200)],
        "t18-17-hecho-capital-y-devolucion": [FakeLLM.llamada_tool(
            "responder", {"respuesta": "Tienen 12 meses de garantía.", "fuentes": ["doc1"]},
            id="x2", tokens_entrada=3000, tokens_salida=400)],
    }

    def fabrica(caso, sink):
        return Orquestador(FakeLLM(llms[caso["id"]]), retriever,
                           Settings(_env_file=None, usar_clasificador_llm=False), trace_sink=sink)

    rep = evals.ejecutar_evals(casos, fabrica, _settings())
    a, b = rep["casos"]
    assert a["accion_ok"] and a["fuentes_ok"] and a["debe_ok"] and a["no_debe_ok"] and not a["conocido"]
    assert not b["accion_ok"] and not b["fuentes_ok"] and not b["debe_ok"] and not b["no_debe_ok"]
    assert b["debe_faltantes"] == ["supervisor"] and b["no_debe_presentes"] == ["12 meses"]
    assert b["conocido"] is True and "falso positivo" in b["motivo_conocido"].lower()
    # Costo a mano: (1000*1 + 200*5)/1e6 = 0.002 y (3000*1 + 400*5)/1e6 = 0.005 USD.
    assert a["costo_usd"] == pytest.approx(0.002) and b["costo_usd"] == pytest.approx(0.005)
    assert rep["global"]["costo_total_usd"] == pytest.approx(0.007)
    assert rep["global"]["n"] == 2 and rep["global"]["accion_acierto"] == 0.5
    assert rep["global"]["fuentes_acierto"] == 0.5 and rep["global"]["caso_ok"] == 0.5
    assert rep["por_categoria"]["politica"]["caso_ok"] == 1.0
    assert rep["por_categoria"]["hecho_critico"]["caso_ok"] == 0.0
    assert "hecho_critico" in rep["por_categoria"] and "pedido" not in rep["por_categoria"]
    md = evals.render_markdown(rep)
    assert "conocido" in md.lower()


def test_los_casos_conocidos_existen_en_el_golden():
    ids = {c["id"] for c in GOLDEN}
    assert set(evals.CASOS_CONOCIDOS) <= ids
    # t18-17 ya no es un fallo conocido: la tabla de hechos se acotó (ADR-009).
    assert "t18-17-hecho-capital-y-devolucion" not in evals.CASOS_CONOCIDOS


# ------------------------------------------------------------------ tope de costo
def test_tope_de_costo_aborta_limpiamente_y_reporta_lo_ejecutado(retriever):
    casos = GOLDEN[:5]
    gastar = lambda: FakeLLM.llamada_tool(
        "responder", {"respuesta": "Hola, ¿en qué te ayudo?", "fuentes": []},
        id="g", tokens_entrada=1_000_000, tokens_salida=0,
    )

    def fabrica(caso, sink):
        return Orquestador(FakeLLM([gastar()]), retriever,
                           Settings(_env_file=None, usar_clasificador_llm=False), trace_sink=sink)

    # Cada caso cuesta 1 USD (1M tokens de entrada); tope 1.5 -> se detiene tras el segundo.
    rep = evals.ejecutar_evals(casos, fabrica, _settings(), max_costo=1.5)
    assert rep["abortado"] is True and rep["casos_ejecutados"] == 2
    assert rep["golden_casos"] == 5 and len(rep["casos"]) == 2
    assert "tope" in rep["motivo_aborto"].lower()
    assert rep["global"]["costo_total_usd"] == pytest.approx(2.0)
    assert "ABORTADO" in evals.render_markdown(rep)


def test_sin_exceder_el_tope_corre_todo(retriever):
    rep = evals.ejecutar_evals(GOLDEN[:3], _fabrica_guiones(retriever), _settings(), max_costo=0.0)
    assert rep["abortado"] is False and rep["casos_ejecutados"] == 3  # costo 0 no supera tope 0


# ------------------------------------------------------------------ reporte
def test_reporte_json_y_md_bien_formados_sin_pii_ni_claves(retriever, tmp_path):
    casos = [c for c in GOLDEN if c["id"] == "vivo2-05-canal-de-contacto"]
    casos[0] = dict(casos[0])
    casos[0]["mensajes"] = [{"role": "user", "content": "Mi correo es ana.perez@correo.com"}]
    texto = ("Puedes escribir a soporte@tiendahogar.example; anotamos ana.perez@correo.com y "
             "+57 300 123 4567.")

    def fabrica(caso, sink):
        llm = FakeLLM([FakeLLM.llamada_tool("responder", {"respuesta": texto, "fuentes": ["doc5"]},
                                            id="p1")])
        return Orquestador(llm, retriever, Settings(_env_file=None, usar_clasificador_llm=False),
                           trace_sink=sink)

    settings = _settings(anthropic_api_key=CLAVE_FALSA)
    rep = evals.ejecutar_evals(casos, fabrica, settings)
    ruta_json, ruta_md = evals.escribir_reporte(rep, tmp_path / "res")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{4}-claude-haiku-4-5-20251001\.json", ruta_json.name)
    assert ruta_md.suffix == ".md" and ruta_md.stem == ruta_json.stem
    contenido_json = ruta_json.read_text(encoding="utf-8")
    datos = json.loads(contenido_json)
    contenido_md = ruta_md.read_text(encoding="utf-8")
    for texto_archivo in (contenido_json, contenido_md):
        assert "ana.perez@correo.com" not in texto_archivo
        assert "4567" not in texto_archivo
        assert CLAVE_FALSA not in texto_archivo and "sk-ant" not in texto_archivo
    assert datos["modelo"] == "claude-haiku-4-5-20251001"
    assert datos["golden_casos"] == 1 and datos["precios"] == {
        "entrada_por_millon_usd": 1.0, "salida_por_millon_usd": 5.0}
    assert "no verificad" in datos["advertencia"].lower()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*", datos["fecha"])
    for titulo in ("Global", "politica", "no verificad", "claude-haiku-4-5-20251001"):
        assert titulo.lower() in contenido_md.lower()
    assert contenido_md.count("|") > 20  # tablas


@pytest.mark.skipif(
    shutil.which("git") is None or not (RAIZ / ".git").exists(),
    reason="requiere git y el repositorio (no están en la imagen Docker)",
)
def test_resultados_no_esta_ignorado_por_git():
    r = subprocess.run(
        ["git", "check-ignore", "-q", "evals/resultados/2026-10-04-1200-modelo.json"],
        cwd=RAIZ, capture_output=True, check=False,
    )
    assert r.returncode == 1  # 1 = no ignorado


# ------------------------------------------------------------------ sin API key
def test_sin_key_se_omite_con_aviso_y_sin_tocar_la_red(monkeypatch, capsys, tmp_path):
    def prohibido(*a, **k):
        raise AssertionError("no debe construir LLM ni retriever sin API key")

    monkeypatch.setattr(evals, "crear_llm", prohibido)
    monkeypatch.setattr(evals, "construir_retriever", prohibido)
    rc = evals.principal(["--salida", str(tmp_path / "res")], settings=_settings())
    salida = capsys.readouterr().out
    assert rc == 0
    assert "omitido: falta ANTHROPIC_API_KEY" in salida
    assert not (tmp_path / "res").exists()


def test_sin_key_tambien_con_settings_por_defecto_en_entorno_aislado(monkeypatch, capsys):
    monkeypatch.setattr(evals, "crear_llm", lambda *a, **k: pytest.fail("sin red"))
    assert evals.principal([]) == 0  # Settings() desde el entorno temporal, sin clave
    assert "omitido: falta ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_con_fabrica_inyectada_corre_y_escribe_el_reporte(retriever, capsys, tmp_path):
    rc = evals.principal(
        ["--salida", str(tmp_path / "res")], settings=_settings(),
        fabrica=_fabrica_guiones(retriever),
    )
    out = capsys.readouterr().out
    assert rc == 0
    archivos = sorted(p.suffix for p in (tmp_path / "res").iterdir())
    assert archivos == [".json", ".md"]
    assert "37" in out and "Global" in out


def test_con_key_arma_llm_anthropic_y_retriever_reales_sin_llamarlos(monkeypatch, retriever):
    """Con clave, `principal` construye LLM y retriever (aquí dobles) y corre con ellos."""
    visto = {}

    def falso_crear_llm(settings):
        visto["provider"] = settings.llm_provider
        return FakeLLM([FakeLLM.vacia()] * 80)

    monkeypatch.setattr(evals, "crear_llm", falso_crear_llm)
    monkeypatch.setattr(evals, "construir_retriever", lambda settings, chunks: retriever)
    fabrica = evals._fabrica_real(_settings(anthropic_api_key=CLAVE_FALSA, llm_provider="anthropic"))
    assert visto["provider"] == "anthropic"
    orq = fabrica(GOLDEN[0], SinkMemoria())
    assert isinstance(orq, Orquestador)
    assert evals.hay_api_key(_settings(anthropic_api_key=CLAVE_FALSA))
    assert not evals.hay_api_key(_settings(anthropic_api_key="   "))
    assert not evals.hay_api_key(_settings())


def test_principal_devuelve_3_si_aborta_por_costo(retriever, capsys, tmp_path):
    def fabrica(caso, sink):
        llm = FakeLLM([FakeLLM.llamada_tool(
            "responder", {"respuesta": "Hola, ¿en qué te ayudo?", "fuentes": []},
            id="c", tokens_entrada=1_000_000)])
        return Orquestador(llm, retriever, Settings(_env_file=None, usar_clasificador_llm=False),
                           trace_sink=sink)

    rc = evals.principal(["--max-costo", "0.5", "--salida", str(tmp_path / "res")],
                         settings=_settings(), fabrica=fabrica)
    assert rc == 3
    assert "tope" in capsys.readouterr().out.lower()
    assert len(list((tmp_path / "res").iterdir())) == 2  # el reporte parcial sí se guarda


# ------------------------------------------------------------------ init.py
def _cargar_init():
    spec = importlib.util.spec_from_file_location("init_harness", RAIZ / "harness" / "init.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _correr_init(monkeypatch, argv):
    init = _cargar_init()
    llamadas: list[tuple[str, ...]] = []

    def falso(py, *args):
        llamadas.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(init, "correr", falso)
    monkeypatch.setattr(sys, "argv", ["init.py", *argv])
    init.main()
    return llamadas


def test_init_sin_full_nunca_ejecuta_los_evals(monkeypatch, capsys):
    llamadas = _correr_init(monkeypatch, [])
    assert llamadas and not any("tiendahogar_agent.evals" in a for args in llamadas for a in args)
    assert "init.py en verde" in capsys.readouterr().out


def test_init_full_ejecuta_el_runner_una_vez(monkeypatch, capsys):
    llamadas = _correr_init(monkeypatch, ["--full"])
    assert sum("tiendahogar_agent.evals" in a for args in llamadas for a in args) == 1
    assert "aún no implementados" not in capsys.readouterr().out


def test_init_full_falla_si_el_runner_falla(monkeypatch):
    init = _cargar_init()

    def falso(py, *args):
        rc = 1 if "tiendahogar_agent.evals" in args else 0
        return SimpleNamespace(returncode=rc, stdout="", stderr="boom")

    monkeypatch.setattr(init, "correr", falso)
    monkeypatch.setattr(sys, "argv", ["init.py", "--full"])
    with pytest.raises(SystemExit) as e:
        init.main()
    assert e.value.code == 1


# ------------------------------------------------------------------ calibración
class _EmbedderTematico:
    """Vector 2D por tema: garantía/envíos/licuadora -> eje 0; todo lo demás -> eje 1."""

    def embed(self, textos):
        salida = []
        for t in textos:
            t = quitar_tildes(t)
            if any(k in t for k in ("garantia", "licuadora", "envios")):
                salida.append([1.0, 0.0])
            else:
                salida.append([0.0, 1.0])
        return salida


def _chunks_sinteticos():
    return [
        Chunk(texto="La garantía cubre defectos de fábrica", doc_id="d1", metadatos={"posicion": 0}),
        Chunk(texto="Los envíos tardan entre dos y tres días", doc_id="d2", metadatos={"posicion": 0}),
    ]


def test_calibracion_sintetica_con_margen_y_casos_del_lado_equivocado():
    dominio = [
        {"id": "d-ok", "categoria": "politica", "pregunta": "garantía de la licuadora"},
        {"id": "d-mal", "categoria": "hecho_critico", "pregunta": "me atienden de forma rara"},
        {"id": "d-ped", "categoria": "pedido", "pregunta": "dónde está lo mío"},  # fuera del margen
    ]
    fuera = [
        {"id": "f-ok", "pregunta": "cómo estará el clima", "limite": False},
        {"id": "f-mal", "pregunta": "garantía de mi vida", "limite": False},
        {"id": "f-lim", "pregunta": "recomiéndame una licuadora", "limite": True},
    ]
    cal = calibracion.calibrar(
        dominio, fuera, _chunks_sinteticos(), _EmbedderTematico(),
        umbral_bm25=0.5, umbral_semantico=0.3,
    )
    por_id = {p["id"]: p for p in cal["preguntas"]}
    assert por_id["d-ok"]["max_coseno"] == pytest.approx(1.0)
    assert por_id["d-ok"]["chunks_relevantes"] == 2  # ambos chunks están en el eje 0
    assert por_id["f-ok"]["max_coseno"] == pytest.approx(0.0, abs=1e-9)
    assert por_id["f-ok"]["max_bm25"] == 0.0 and por_id["f-ok"]["chunks_relevantes"] == 0
    assert por_id["d-mal"]["chunks_relevantes"] == 0
    assert por_id["f-lim"]["limite"] is True and por_id["f-lim"]["chunks_relevantes"] == 2
    equivocados = {c["id"]: c["motivo"] for c in cal["lado_equivocado"]}
    assert set(equivocados) == {"d-mal", "f-mal"}  # límite y pedido no cuentan como error
    assert "sin" in equivocados["d-mal"] and "fuera" in equivocados["f-mal"]
    sem = cal["resumen"]["coseno"]
    assert sem["fuera_de_dominio"]["max"] == pytest.approx(1.0)  # f-mal (la límite no cuenta)
    assert sem["en_dominio"]["min"] == pytest.approx(0.0, abs=1e-9)  # d-mal
    assert sem["en_dominio"]["mediana"] == pytest.approx(0.5)
    assert sem["margen"] == pytest.approx(-1.0)  # negativo: las distribuciones se solapan
    assert cal["resumen"]["umbral_semantico"]["margen_inferior"] == pytest.approx(-0.3)
    assert cal["resumen"]["umbral_semantico"]["margen_superior"] == pytest.approx(-0.7)
    assert cal["resumen"]["n"] == {"en_dominio": 2, "fuera_de_dominio": 2, "limite": 1, "otras": 1}


def test_calibracion_margen_positivo_cuando_se_separan():
    dominio = [{"id": "d1", "categoria": "politica", "pregunta": "garantía de la licuadora"}]
    fuera = [{"id": "f1", "pregunta": "lluvia hoy", "limite": False}]
    cal = calibracion.calibrar(dominio, fuera, _chunks_sinteticos(), _EmbedderTematico(),
                               umbral_bm25=0.5, umbral_semantico=0.3)
    assert cal["resumen"]["coseno"]["margen"] == pytest.approx(1.0)
    assert cal["lado_equivocado"] == []
    assert cal["resumen"]["umbral_semantico"]["margen_inferior"] == pytest.approx(0.7)
    assert cal["resumen"]["umbral_semantico"]["margen_superior"] == pytest.approx(0.3)


def test_calibracion_consistente_con_la_regla_del_retriever(retriever):
    """`chunks_relevantes` coincide con lo que el retriever real devolvería (sin top_k)."""
    chunks = FileSystemDocumentSource(DOCS).cargar()
    preguntas = [{"id": "g", "categoria": "politica",
                  "pregunta": "¿Cuánto dura la garantía de mi licuadora?"}]
    emb = FakeEmbedder()
    cal = calibracion.calibrar(preguntas, [], chunks, emb, umbral_bm25=0.5, umbral_semantico=0.3)
    completo = Retriever(chunks, IndiceLexico(chunks), emb, InMemoryVectorStore(),
                         top_k=len(chunks), umbral_bm25=0.5, umbral_semantico=0.3)
    esperado = len(completo.recuperar(preguntas[0]["pregunta"]))
    assert cal["preguntas"][0]["chunks_relevantes"] == esperado


def test_preguntas_fuera_de_dominio_bien_formadas():
    datos = json.loads((RAIZ / "evals" / "preguntas_fuera_de_dominio.json").read_text(encoding="utf-8"))
    assert 18 <= len(datos) <= 30
    ids = [d["id"] for d in datos]
    assert len(set(ids)) == len(ids)
    for d in datos:
        assert set(d) == {"id", "tema", "pregunta", "limite"}
        assert d["pregunta"].strip() and isinstance(d["limite"], bool)
    assert len({d["pregunta"] for d in datos}) == len(datos)
    assert sum(d["limite"] for d in datos) >= 2
    assert len({d["tema"] for d in datos}) >= 7
    assert not re.search(r"sk-|api[_-]?key", json.dumps(datos), re.IGNORECASE)


def test_calibracion_escribe_json_y_md(tmp_path):
    dominio = [{"id": "d1", "categoria": "politica", "pregunta": "garantía de la licuadora"}]
    fuera = [{"id": "f1", "pregunta": "lluvia hoy", "limite": False}]
    cal = calibracion.calibrar(dominio, fuera, _chunks_sinteticos(), _EmbedderTematico(),
                               umbral_bm25=0.5, umbral_semantico=0.3)
    rj, rm = calibracion.escribir_calibracion(cal, tmp_path)
    assert re.fullmatch(r"calibracion-\d{4}-\d{2}-\d{2}\.json", rj.name) and rm.suffix == ".md"
    assert json.loads(rj.read_text(encoding="utf-8"))["resumen"]["coseno"]["margen"] == 1.0
    assert "margen" in rm.read_text(encoding="utf-8").lower()


def test_golden_preguntas_de_calibracion_toman_el_ultimo_mensaje_user():
    preguntas = calibracion.preguntas_del_golden(GOLDEN)
    assert len(preguntas) == 37
    por_id = {p["id"]: p for p in preguntas}
    caso = next(c for c in GOLDEN if c["id"] == "t18-01-lavadora-defecto-multiturno")
    assert por_id[caso["id"]]["pregunta"] == caso["mensajes"][-1]["content"]
    assert por_id[caso["id"]]["categoria"] == "politica"
