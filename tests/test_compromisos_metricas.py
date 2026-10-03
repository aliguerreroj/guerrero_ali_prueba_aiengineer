"""Métrica medible del detector de compromisos sobre un conjunto FIJO de frases.

El conjunto (tests/data/compromisos_eval.json) no se afina frase por frase: sirve
para medir. Criterio de aceptación: detección >= 90 % y falsos positivos <= 5 %.
"""

from __future__ import annotations

import json
from pathlib import Path

from tiendahogar_agent.guardrail_compromisos import detectar_compromisos

RUTA = Path(__file__).parent / "data" / "compromisos_eval.json"
UMBRAL_DETECCION = 0.90
UMBRAL_FALSOS_POSITIVOS = 0.05


def _medir() -> dict:
    datos = json.loads(RUTA.read_text(encoding="utf-8"))
    comp = [d for d in datos if d["esperado"] == "compromiso"]
    leg = [d for d in datos if d["esperado"] == "legitima"]
    detectados = [d for d in comp if detectar_compromisos(d["frase"])]
    falsos = [d for d in leg if detectar_compromisos(d["frase"])]
    return {
        "comp": comp, "leg": leg,
        "perdidos": [d["frase"] for d in comp if d not in detectados],
        "aciertos": len(detectados), "falsos": falsos,
    }


def test_conjunto_bien_formado():
    datos = json.loads(RUTA.read_text(encoding="utf-8"))
    assert all(set(d) == {"frase", "esperado", "familia"} for d in datos)
    assert {d["esperado"] for d in datos} == {"compromiso", "legitima"}
    assert len({d["frase"] for d in datos}) == len(datos)
    assert len(datos) >= 100


def test_metricas_detector_compromisos():
    m = _medir()
    n_c, n_l = len(m["comp"]), len(m["leg"])
    deteccion = m["aciertos"] / n_c
    fp = len(m["falsos"]) / n_l
    resumen = (
        f"deteccion={m['aciertos']}/{n_c} ({deteccion:.1%}); "
        f"falsos_positivos={len(m['falsos'])}/{n_l} ({fp:.1%}); "
        f"perdidos={m['perdidos']}; falsos={[d['frase'] for d in m['falsos']]}"
    )
    print(resumen)
    assert deteccion >= UMBRAL_DETECCION, resumen
    assert fp <= UMBRAL_FALSOS_POSITIVOS, resumen
