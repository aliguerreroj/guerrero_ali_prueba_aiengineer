"""Ayuda para los topes de tiempo de los tests de rendimiento.

Los topes detectan regex de coste cuadrático (que tardarían órdenes de magnitud más). Bajo
integración continua (variable de entorno `CI`) los runners compartidos son más lentos, así que
el tope se multiplica por `FACTOR_CI`; fuera de CI no cambia nada.
"""

from __future__ import annotations

import os

FACTOR_CI = 5


def tope(segundos: float, entorno: dict[str, str] | None = None) -> float:
    """Devuelve el tope en segundos, multiplicado por FACTOR_CI si la variable CI está presente."""
    env = os.environ if entorno is None else entorno
    return segundos * FACTOR_CI if env.get("CI") else segundos
