"""Enmascarado determinista de PII (correos, tarjetas y teléfonos).

Se aplica antes de escribir logs y antes de enviar texto al LLM.

Frontera de decisiones:
- Correo: parte local y etiquetas de dominio acotadas (sin ReDoS); no distingue mayúsculas.
- Tarjeta: 13 a 19 dígitos con espacios o guiones como separadores, y solo si pasa Luhn.
  Un candidato que no pasa Luhn no se enmascara como tarjeta; se reintenta como teléfono
  sobre sus propios dígitos (así "ORD-1001 3001234567" sigue enmascarando el teléfono).
  Con separadores, cada grupo debe tener 4 a 6 dígitos (4-4-4-4, 4-6-5...), así un id o
  número corto seguido de un teléfono no se confunde con una tarjeta aunque pase Luhn.
- NFKC (por carácter, con mapa de índices) y tab como espacio solo para detectar; el texto
  devuelto conserva intacto todo lo que no es PII (ver `_normalizar_con_mapa`).
- Separadores de tarjeta: espacio (hasta 2), '.', '-' o '/'; el último grupo puede ser corto
  (1 a 6 dígitos) y los anteriores deben tener 4 a 6.
- Limitaciones documentadas: teléfonos no colombianos sin '+'/00 no se enmascaran (decisión
  de diseño); "3000000000 pesos" es ambiguo y se enmascara como teléfono; un id pegado a una
  tarjeta sin separador no se detecta; un correo sin TLD no se enmascara.
- Teléfono: móvil colombiano = 10 dígitos que empiezan por 3 (con o sin 57/0/+57/0057,
  indicativo entre paréntesis admitido); internacional también con prefijo 00;
  fijo = indicativo 60X + 7 dígitos; internacional = '+' seguido de 8 a 15 dígitos
  (admite espacios, puntos, guiones y paréntesis).
- Nunca se tocan ids de pedido (ORD-####), montos, fechas, años ni números sueltos de
  4 a 9 dígitos: ningún patrón los cubre.
- Orden: correo, luego una sola pasada con (internacional | tarjeta | teléfono).
- Los marcadores solo tienen letras y corchetes, por lo que la función es idempotente.
"""

import re
import unicodedata

CORREO = "[CORREO]"
TELEFONO = "[TELEFONO]"
TARJETA = "[TARJETA]"

_RE_CORREO = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@"
    r"[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){1,8}"
)

_SEP = r"[ .\-()]{0,2}"
_INTL = (
    r"(?<![\w+$])(?:\+\d(?:" + _SEP + r"\d){7,14}|00[1-9](?:" + _SEP + r"\d){7,13})(?!\d)"
)
_TEL_COL = (
    r"(?<![\d$])(?<!\d[.,])"
    r"(?:(?:57|0)[ .\-]?)?"
    r"(?:\(?3\d{2}\)?[ .\-]?\d{3}[ .\-]?\d{4}|\(?60\d\)?[ .\-]?\d{3}[ .\-]?\d{4})"
    r"(?!\d)(?![.,]\d)"
)
_RE_TEL = re.compile(_INTL + "|" + _TEL_COL)

_RE_PII = re.compile(
    r"(?P<intl>" + _INTL + ")"
    r"|(?P<tarjeta>(?<![\d$])\d(?:[ .\-/]{0,2}\d){12,18}(?!\d))"
    r"|(?P<tel>" + _TEL_COL + ")"
)
_RE_SEP_TARJETA = re.compile(r"[ .\-/]+")


def _normalizar_con_mapa(texto: str) -> tuple[str, list[int]]:
    """NFKC por carácter (y tab como espacio) con mapa a los índices del original.

    La normalización solo sirve para DETECTAR (dígitos de ancho completo, NBSP...). Cada
    carácter normalizado apunta al índice del carácter original del que proviene, así los
    tramos detectados se reemplazan en el texto original y lo que no es PII queda intacto.
    """
    partes: list[str] = []
    mapa: list[int] = []
    for i, c in enumerate(texto):
        n = " " if c == "	" else unicodedata.normalize("NFKC", c)
        partes.append(n)
        mapa.extend([i] * len(n))
    return "".join(partes), mapa


def _sustituir(texto: str, patron: re.Pattern, fn) -> str:
    """Detecta sobre el texto normalizado y reemplaza los tramos en el original."""
    norm, mapa = _normalizar_con_mapa(texto)
    salida: list[str] = []
    pos = 0
    for m in patron.finditer(norm):
        if m.end() == m.start():
            continue
        ini, fin = mapa[m.start()], mapa[m.end() - 1] + 1
        salida.append(texto[pos:ini])
        salida.append(fn(m))
        pos = fin
    salida.append(texto[pos:])
    return "".join(salida)


def _luhn(digitos: str) -> bool:
    if not 13 <= len(digitos) <= 19:
        return False
    total = 0
    for i, c in enumerate(reversed(digitos)):
        n = int(c)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def enmascarar_pii_con_conteos(texto: str) -> tuple[str, dict[str, int]]:
    """Enmascara PII y devuelve (texto, conteos por tipo)."""
    conteos = {"correo": 0, "telefono": 0, "tarjeta": 0}
    if not texto:
        return "", conteos

    def _correo(_m: re.Match) -> str:
        conteos["correo"] += 1
        return CORREO

    def _tel_simple(_m: re.Match) -> str:
        conteos["telefono"] += 1
        return TELEFONO

    def _pii(m: re.Match) -> str:
        if m.group("intl") or m.group("tel"):
            return _tel_simple(m)
        candidato = m.group("tarjeta")
        grupos = _RE_SEP_TARJETA.split(candidato)
        grupos_ok = len(grupos) == 1 or (
            all(4 <= len(g) <= 6 for g in grupos[:-1]) and 1 <= len(grupos[-1]) <= 6
        )
        if grupos_ok and _luhn("".join(grupos)):
            conteos["tarjeta"] += 1
            return TARJETA
        return _RE_TEL.sub(_tel_simple, candidato)

    texto = _sustituir(texto, _RE_CORREO, _correo)
    return _sustituir(texto, _RE_PII, _pii), conteos


def enmascarar_pii(texto: str) -> str:
    """Enmascara correos, tarjetas (Luhn) y teléfonos con marcadores estables."""
    return enmascarar_pii_con_conteos(texto)[0]
