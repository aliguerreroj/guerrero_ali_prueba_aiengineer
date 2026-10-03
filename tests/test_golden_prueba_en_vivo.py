"""Valida solo el ESQUEMA del golden set de la prueba en vivo (no que el agente lo cumpla; eso es T18)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RUTA = Path(__file__).parent / "data" / "golden_prueba_en_vivo.json"
CAMPOS = {"id", "pregunta", "accion_esperada", "fuentes_esperadas", "notas", "origen"}
CAMPOS_OPCIONALES = {"historial"}
ORIGENES = {"prueba_en_vivo", "prueba_en_vivo_2"}
ACCIONES = {"responder", "escalar", "pedir_dato"}
FUENTES_VALIDAS = {"doc1", "doc2", "doc3", "doc4", "doc5", "pedidos"}


@pytest.fixture(scope="module")
def casos():
    return json.loads(RUTA.read_text(encoding="utf-8"))


def test_hay_doce_casos_con_ids_unicos(casos):
    assert isinstance(casos, list) and len(casos) == 12
    assert len({c["id"] for c in casos}) == 12
    assert sum(c["origen"] == "prueba_en_vivo" for c in casos) == 10


def test_cada_caso_tiene_los_campos_y_tipos_correctos(casos):
    for c in casos:
        assert CAMPOS <= set(c) <= CAMPOS | CAMPOS_OPCIONALES, c.get("id")
        for t in c.get("historial", []):
            assert set(t) == {"role", "content"} and t["role"] in ("user", "assistant"), c["id"]
            assert isinstance(t["content"], str) and t["content"].strip(), c["id"]
        for campo in ("id", "pregunta", "notas"):
            assert isinstance(c[campo], str) and c[campo].strip(), (c["id"], campo)
        assert c["origen"] in ORIGENES, c["id"]
        assert c["accion_esperada"] in ACCIONES, c["id"]
        assert isinstance(c["fuentes_esperadas"], list), c["id"]
        assert set(c["fuentes_esperadas"]) <= FUENTES_VALIDAS, c["id"]


def test_coherencia_entre_accion_y_fuentes(casos):
    for c in casos:
        if c["accion_esperada"] in ("escalar", "pedir_dato"):
            assert c["fuentes_esperadas"] == [], c["id"]


def test_cubre_las_tres_acciones(casos):
    assert {c["accion_esperada"] for c in casos} == ACCIONES


def test_no_contiene_secretos(casos):
    texto = json.dumps(casos, ensure_ascii=False)
    assert not re.search(r"sk-|api[_-]?key", texto, re.IGNORECASE)


@pytest.mark.parametrize("id_caso", ["vivo-03-queja-trato-empleado", "vivo-04-queja-trato-vendedor"])
def test_quejas_de_trato_escalan_por_reglas_sin_llm(casos, id_caso):
    """Regresión: ambas redacciones escalan en la primera capa (determinista), sin clasificador."""
    from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO, evaluar_entrada

    caso = next(c for c in casos if c["id"] == id_caso)
    r = evaluar_entrada(caso["pregunta"], 500)
    assert r.escalar and r.categoria == "queja_trato" and r.canal == CANAL_ESCALAMIENTO
