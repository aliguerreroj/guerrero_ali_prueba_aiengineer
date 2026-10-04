"""Golden set de casos trampa (T18): `evals/golden_set.json`.

Dos grupos de pruebas, ambos deterministas (FakeLLM, sin red ni API key):

1. Esquema y coherencia del archivo (tipos, ids, categorías, acciones, fuentes, cobertura de los
   criterios de T18, mensajes bien formados, sin secretos).
2. Determinismo del orquestador: cada caso se ejecuta con el orquestador real y el retriever
   léxico. Los casos que el guardrail de entrada escala por reglas no llevan guion (se usa un
   LLM vacío y sale la plantilla de respaldo); los demás usan el guion de `golden_guiones.py`.

IMPORTANTE: comprobar `debe_contener`/`no_debe_contener` contra la respuesta guionada valida el
golden y los guiones (que las expectativas son consistentes entre sí y con el sistema), NO mide
al LLM real. La medición con el LLM real es T19.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from golden_guiones import GUIONES

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO, evaluar_entrada
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.matching import alternativas, faltantes, normalizar, presentes
from tiendahogar_agent.orquestador import Orquestador
from tiendahogar_agent.retriever import Retriever

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "evals" / "golden_set.json"
DOCS = RAIZ / "data" / "docs"

CAMPOS_OBLIGATORIOS = {
    "id", "categoria", "origen", "mensajes", "accion_esperada", "fuentes_esperadas",
    "debe_contener", "no_debe_contener", "notas",
}
CAMPOS_OPCIONALES = {"criterio_t18"}
ACCIONES = {"responder", "escalar", "pedir_dato"}
CATEGORIAS = {
    "politica", "pedido", "escalamiento", "fuera_de_alcance", "manipulacion", "hecho_critico",
}
ORIGENES = {"prueba_en_vivo", "prueba_en_vivo_2", "prueba_en_vivo_3", "t18"}
# Un criterio_t18 por cada criterio de la tarea (el primero, el archivo versionado, se valida aparte).
CRITERIOS_T18 = {
    "lavadora_defecto_30_dias", "envio_internacional", "liquidacion", "ord_1002_entregado",
    "montos", "mezcla_temas", "producto_no_listado", "pedido_inexistente", "prompt_injection",
}
IDS_MIGRADOS = [
    "vivo-01-licuadora-12-meses", "vivo-02-fuera-de-alcance-mundial",
    "vivo-03-queja-trato-empleado", "vivo-04-queja-trato-vendedor",
    "vivo-05-estado-pedido-existente", "vivo-06-estado-pedido-sin-id",
    "vivo-07-reembolso-sobre-umbral", "vivo-08-devolucion-pasados-30-dias",
    "vivo-09-envio-otras-ciudades", "vivo-10-promesa-de-reembolso",
    "vivo2-01-pedido-inexistente", "vivo2-02-seguimiento-urgente",
    "vivo2-03-reportar-trato-empleado", "vivo2-04-atendio-pesimo-vendedor",
    "vivo2-05-canal-de-contacto", "vivo2-06-microondas-no-listado", "vivo2-07-envio-capital",
    "vivo2-08-lavadora-devolucion-defecto", "vivo3-01-lavadora-defecto-condicional",
]
FUENTES_DOC = {"doc1", "doc2", "doc3", "doc4", "doc5"}


def _cargar() -> list[dict]:
    return json.loads(RUTA.read_text(encoding="utf-8"))


CASOS = _cargar()
IDS = [c["id"] for c in CASOS]


@pytest.fixture(scope="module")
def casos():
    return CASOS


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


# ------------------------------------------------------------------ esquema
def test_archivo_versionado_y_unificado():
    assert RUTA.is_file()
    assert not (RAIZ / "tests" / "data" / "golden_prueba_en_vivo.json").exists()


def test_al_menos_28_casos_con_ids_unicos(casos):
    assert isinstance(casos, list) and len(casos) >= 28
    assert len(set(IDS)) == len(IDS)
    assert set(IDS_MIGRADOS) <= set(IDS)
    assert sum(c["origen"].startswith("prueba_en_vivo") for c in casos) == len(IDS_MIGRADOS)


def test_cada_caso_tiene_los_campos_y_tipos_correctos(casos):
    for c in casos:
        assert CAMPOS_OBLIGATORIOS <= set(c) <= CAMPOS_OBLIGATORIOS | CAMPOS_OPCIONALES, c.get("id")
        for campo in ("id", "notas"):
            assert isinstance(c[campo], str) and c[campo].strip(), (c["id"], campo)
        assert c["origen"] in ORIGENES, c["id"]
        assert c["categoria"] in CATEGORIAS, c["id"]
        assert c["accion_esperada"] in ACCIONES, c["id"]
        assert isinstance(c["fuentes_esperadas"], list), c["id"]
        assert len(set(c["fuentes_esperadas"])) == len(c["fuentes_esperadas"]), c["id"]
        if "criterio_t18" in c:
            assert c["criterio_t18"] in CRITERIOS_T18, c["id"]


def test_mensajes_bien_formados(casos):
    for c in casos:
        m = c["mensajes"]
        assert isinstance(m, list) and m, c["id"]
        for t in m:
            assert set(t) == {"role", "content"}, c["id"]
            assert t["role"] in ("user", "assistant"), c["id"]
            assert isinstance(t["content"], str) and t["content"].strip(), c["id"]
        assert m[-1]["role"] == "user", c["id"]
        roles = [t["role"] for t in m]
        # Alterna user/assistant y empieza con el usuario (que es quien escribe el último turno).
        assert roles == ["user", "assistant"] * (len(roles) // 2) + ["user"] * (len(roles) % 2), c["id"]
        assert len(roles) % 2 == 1, c["id"]


def test_fuentes_existen_en_docs_o_son_pedidos(casos):
    en_disco = {p.name.split("_")[0] for p in DOCS.glob("doc*.md")}
    assert en_disco == FUENTES_DOC
    for c in casos:
        assert set(c["fuentes_esperadas"]) <= FUENTES_DOC | {"pedidos"}, c["id"]
        assert set(c["fuentes_esperadas"]) <= en_disco | {"pedidos"}, c["id"]


def test_coherencia_accion_fuentes_categoria(casos):
    """Reglas del repo: escalar/pedir_dato no citan fuentes; cada categoría fija su acción."""
    for c in casos:
        f, acc, cat = c["fuentes_esperadas"], c["accion_esperada"], c["categoria"]
        if acc in ("escalar", "pedir_dato"):
            assert f == [], c["id"]
        if cat in ("escalamiento", "manipulacion"):
            assert acc == "escalar" and f == [], c["id"]
        elif cat == "fuera_de_alcance":
            assert acc == "responder" and f == [], c["id"]
        elif cat == "pedido":
            assert acc in ("responder", "pedir_dato"), c["id"]
            assert set(f) <= {"pedidos"}, c["id"]
            assert (acc == "responder") == (f == ["pedidos"]), c["id"]
        else:  # politica | hecho_critico
            assert acc == "responder" and f and set(f) <= FUENTES_DOC, c["id"]


def test_cobertura_de_categorias_acciones_y_criterios(casos):
    assert {c["categoria"] for c in casos} == CATEGORIAS
    assert {c["accion_esperada"] for c in casos} == ACCIONES
    assert {c["criterio_t18"] for c in casos if "criterio_t18" in c} == CRITERIOS_T18
    assert sum(len(c["mensajes"]) > 1 for c in casos) >= 4
    assert sum(c["categoria"] == "fuera_de_alcance" for c in casos) >= 5


def test_montos_cubren_umbral_letras_y_formato(casos):
    ultimos = {c["id"]: c["mensajes"][-1]["content"] for c in casos if c.get("criterio_t18") == "montos"}
    texto = " ".join(ultimos.values())
    assert "$500" in texto and "$501" in texto and "ochocientos" in texto and "$1.200" in texto


def test_listas_de_cadenas_no_vacias_y_normalizables(casos):
    """Las cadenas admiten alternativas con «|» (matching.py); ninguna puede quedar vacía."""
    for c in casos:
        for campo in ("debe_contener", "no_debe_contener"):
            lista = c[campo]
            assert isinstance(lista, list) and lista, (c["id"], campo)
            for s in lista:
                assert isinstance(s, str) and alternativas(s), (c["id"], campo)
            assert len({normalizar(s) for s in lista}) == len(lista), (c["id"], campo)
        # Lo que debe estar no puede estar a la vez prohibido (ni como alternativa).
        debe = {a for s in c["debe_contener"] for a in alternativas(s)}
        no_debe = {a for s in c["no_debe_contener"] for a in alternativas(s)}
        assert not debe & no_debe, c["id"]


def test_no_contiene_secretos(casos):
    texto = json.dumps(casos, ensure_ascii=False)
    assert not re.search(r"sk-|api[_-]?key|bearer\s", texto, re.IGNORECASE)


def test_guiones_corresponden_a_casos_que_no_escalan(casos):
    sin_guion = {c["id"] for c in casos if c["id"] not in GUIONES}
    escalan = {c["id"] for c in casos if c["accion_esperada"] == "escalar"}
    assert sin_guion == escalan
    assert set(GUIONES) <= set(IDS)


# ------------------------------------------------------------------ determinismo
def _procesar(retriever, caso, guion):
    if guion is None:  # escalamiento por reglas: el LLM no aporta texto, sale la plantilla
        llm = FakeLLM([FakeLLM.vacia()])
        usar_clasificador = False
    else:
        llm = FakeLLM(list(guion.respuestas))
        usar_clasificador = guion.clasificador
    orq = Orquestador(llm, retriever, Settings(usar_clasificador_llm=usar_clasificador))
    mensajes = caso["mensajes"]
    return orq.procesar(mensajes[-1]["content"], historial=mensajes[:-1] or None), llm


@pytest.mark.parametrize("caso", CASOS, ids=IDS)
def test_orquestador_da_la_accion_y_fuentes_esperadas(retriever, caso):
    guion = GUIONES.get(caso["id"])
    ultimo = caso["mensajes"][-1]["content"]
    decision = evaluar_entrada(ultimo, Settings().umbral_reembolso)
    assert decision.escalar == (caso["accion_esperada"] == "escalar"), "reglas de entrada"
    r, llm = _procesar(retriever, caso, guion)
    assert r.accion == caso["accion_esperada"], r.respuesta
    assert r.fuentes == caso["fuentes_esperadas"], r.respuesta
    assert (r.canal == CANAL_ESCALAMIENTO) == (caso["accion_esperada"] == "escalar")
    if guion is not None:
        assert not llm._cola, "el guion debe consumirse completo"
    assert not faltantes(r.respuesta, caso["debe_contener"]), (caso["id"], r.respuesta)
    assert not presentes(r.respuesta, caso["no_debe_contener"]), (caso["id"], r.respuesta)


@pytest.mark.xfail(
    strict=True,
    reason="Falso positivo conocido de la tabla de hechos (ADR-009): «30 días» junto a «capital» "
    "se lee como plazo de envío y se bloquea por hecho_incorrecto. Al corregirlo, quitar el xfail.",
)
def test_falso_positivo_redaccion_natural_capital_y_30_dias(retriever):
    caso = next(c for c in CASOS if c["id"] == "t18-17-hecho-capital-y-devolucion")
    texto = "Si recibes tu pedido en la capital, tienes 30 días para devolverlo."
    llm = FakeLLM([
        FakeLLM.llamada_tool("responder", {"respuesta": texto, "fuentes": ["doc2", "doc3"]}, id="fp1"),
        FakeLLM.llamada_tool("responder", {"respuesta": texto, "fuentes": ["doc2", "doc3"]}, id="fp2"),
    ])
    orq = Orquestador(llm, retriever, Settings(usar_clasificador_llm=False))
    r = orq.procesar(caso["mensajes"][-1]["content"])
    assert r.accion == "responder" and r.fuentes == ["doc2", "doc3"]
