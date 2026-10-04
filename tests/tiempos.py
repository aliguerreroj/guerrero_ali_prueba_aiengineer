"""Ayuda para los topes de tiempo de los tests de rendimiento.

Los topes solo detectan comportamiento cuadrático o con backtracking catastrófico (que tardaría
órdenes de magnitud más), no miden velocidad. Por eso son generosos siempre, en local y en CI:
el tope base se multiplica por `FACTOR_TOPE` para que una máquina lenta o cargada no los rompa.
"""

from __future__ import annotations

FACTOR_TOPE = 10


def tope(segundos: float) -> float:
    """Devuelve el tope en segundos: el tope base multiplicado por FACTOR_TOPE."""
    return segundos * FACTOR_TOPE
