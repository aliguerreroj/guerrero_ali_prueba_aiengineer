"""Matching de `debe_contener` / `no_debe_contener` del golden set (T19).

Un único módulo para el runner de evals y para `tests/test_golden_set.py`.

- Sin tildes, sin mayúsculas y con los espacios (también saltos de línea) colapsados.
- Alternativas con «|» dentro de la cadena: «numero de pedido|numero del pedido» se cumple si
  aparece cualquiera de las dos (como subcadena del texto normalizado).
- Una alternativa vacía se ignora (una barra suelta nunca hace que todo coincida).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from tiendahogar_agent.texto import quitar_tildes

SEPARADOR_ALTERNATIVAS = "|"
_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados."""
    return _ESPACIOS.sub(" ", quitar_tildes(texto or "")).strip()


def alternativas(patron: str) -> list[str]:
    """Alternativas normalizadas (sin vacías) de un patrón con «|»."""
    partes = (normalizar(p) for p in (patron or "").split(SEPARADOR_ALTERNATIVAS))
    return [p for p in partes if p]


def contiene(texto: str, patron: str) -> bool:
    """True si el texto contiene alguna de las alternativas del patrón."""
    base = normalizar(texto)
    return any(a in base for a in alternativas(patron))


def faltantes(texto: str, patrones: Iterable[str]) -> list[str]:
    """Patrones de `debe_contener` que NO aparecen en el texto."""
    return [p for p in patrones if not contiene(texto, p)]


def presentes(texto: str, patrones: Iterable[str]) -> list[str]:
    """Patrones de `no_debe_contener` que SÍ aparecen en el texto."""
    return [p for p in patrones if contiene(texto, p)]
