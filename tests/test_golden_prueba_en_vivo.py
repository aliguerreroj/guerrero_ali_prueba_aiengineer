"""Regresiones de las pruebas en vivo dentro del golden set unificado (`evals/golden_set.json`).

El esquema completo del golden lo valida `test_golden_set.py`; aquí se conserva lo propio de los
19 casos migrados de las pruebas en vivo (ids, orígenes y quejas de trato por reglas).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "evals" / "golden_set.json"
ORIGENES = {"prueba_en_vivo": 10, "prueba_en_vivo_2": 8, "prueba_en_vivo_3": 1}


@pytest.fixture(scope="module")
def casos():
    return json.loads(RUTA.read_text(encoding="utf-8"))


def test_los_diecinueve_casos_migrados_conservan_ids_y_origen(casos):
    vivos = [c for c in casos if c["origen"].startswith("prueba_en_vivo")]
    assert len(vivos) == 19 and len({c["id"] for c in vivos}) == 19
    for origen, cantidad in ORIGENES.items():
        assert sum(c["origen"] == origen for c in vivos) == cantidad
    assert all(c["id"].startswith("vivo") for c in vivos)


def test_los_casos_migrados_tienen_pregunta_final_del_usuario(casos):
    for c in casos:
        if c["origen"].startswith("prueba_en_vivo"):
            assert c["mensajes"][-1]["role"] == "user" and c["mensajes"][-1]["content"].strip()


def test_el_seguimiento_urgente_conserva_su_historial(casos):
    caso = next(c for c in casos if c["id"] == "vivo2-02-seguimiento-urgente")
    assert [m["role"] for m in caso["mensajes"]] == ["user", "assistant", "user"]
    assert "ORD-1003" in caso["mensajes"][0]["content"]


@pytest.mark.parametrize("id_caso", ["vivo-03-queja-trato-empleado", "vivo-04-queja-trato-vendedor"])
def test_quejas_de_trato_escalan_por_reglas_sin_llm(casos, id_caso):
    """Regresión: ambas redacciones escalan en la primera capa (determinista), sin clasificador."""
    from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO, evaluar_entrada

    caso = next(c for c in casos if c["id"] == id_caso)
    r = evaluar_entrada(caso["mensajes"][-1]["content"], 500)
    assert r.escalar and r.categoria == "queja_trato" and r.canal == CANAL_ESCALAMIENTO
