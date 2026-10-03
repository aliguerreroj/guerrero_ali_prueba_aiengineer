"""Normalización de texto en español para la búsqueda léxica (funciones puras).

Criterio: minúsculas, descomposición NFKD y eliminación de marcas combinantes
(tildes y diéresis). Consecuencia deliberada: la «ñ» también pierde su tilde
y queda como «n» ("año" -> "ano"), de modo que las consultas escritas sin
tildes ni eñes coinciden con los documentos. Luego se tokeniza por secuencias
alfanuméricas, se descartan stopwords (lista corta y propia) y se aplica el
stemmer Snowball español.
"""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

# Lista corta y propia (ya sin tildes). Solo palabras funcionales muy frecuentes
# que, sin filtrar, producirían coincidencias espurias en consultas fuera de dominio.
STOPWORDS = frozenset(
    {
    "a", "al", "algo", "ante", "con", "cual", "cuales", "cuando", "cuanto", "cuantos", "de",
    "del", "donde", "e", "el", "ella", "ellas", "ello", "ellos", "en", "entre", "es", "esa",
    "ese", "eso", "esta", "este", "esto", "ha", "han", "hacen", "hace", "la", "las", "le",
    "les", "lo", "los", "me", "mi", "mis", "mas", "muy", "no", "nos", "o", "para", "pero",
    "por", "que", "quien", "quienes", "se", "si", "sin", "su", "sus", "te", "tu", "tus", "un",
    "una", "uno", "unas", "unos", "y", "ya", "yo",
    }
)

_PALABRAS = re.compile(r"[a-z0-9]+")
_stemmer = snowballstemmer.stemmer("spanish")


def quitar_tildes(texto: str) -> str:
    """Minúsculas y sin marcas diacríticas (la ñ queda como n)."""
    descompuesto = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def normalizar(texto: str | None) -> list[str]:
    """Texto -> lista de tokens normalizados (vacía si no hay contenido)."""
    if not texto:
        return []
    palabras = _PALABRAS.findall(quitar_tildes(texto))
    return [_stemmer.stemWord(p) for p in palabras if p not in STOPWORDS]


_CATEGORIAS_INVISIBLES = frozenset({"Cf", "Cc", "Zs", "Zl", "Zp"})


def es_vacio_visible(texto: str | None) -> bool:
    """True si no hay ningún carácter visible (solo espacios, controles o invisibles)."""
    return all(unicodedata.category(c) in _CATEGORIAS_INVISIBLES for c in (texto or ""))
