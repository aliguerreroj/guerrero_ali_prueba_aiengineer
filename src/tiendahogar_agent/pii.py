"""Enmascarado determinista de PII (correos, tarjetas y teléfonos).

Se aplica antes de escribir logs y antes de enviar texto al LLM.

Frontera de decisiones:
- Correo: parte local y etiquetas de dominio acotadas (sin ReDoS); no distingue mayúsculas.
- Tarjeta: 13 a 19 dígitos con espacios o guiones como separadores, y solo si pasa Luhn.
  Un candidato que no pasa Luhn no se enmascara como tarjeta; se reintenta como teléfono
  sobre sus propios dígitos (así "ORD-1001 3001234567" sigue enmascarando el teléfono).
  Con separadores, cada grupo debe tener 4 a 6 dígitos (4-4-4-4, 4-6-5...), así un id o
  número corto seguido de un teléfono no se confunde con una tarjeta aunque pase Luhn.
- Texto de entrada normalizado con NFKC y tab como espacio (ver `_normalizar`).
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


def _normalizar(texto: str) -> str:
    """NFKC (dígitos de ancho completo, NBSP) y tabuladores como espacio.

    Se aplica de forma uniforme a todo el texto devuelto; NFKC también altera texto no PII
    (p. ej. ligaduras, fracciones, formas compatibles), lo cual es aceptable aquí y es
    idempotente.
    """
    return unicodedata.normalize("NFKC", texto).replace("	", " ")


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

    texto = _normalizar(texto)
    return _RE_PII.sub(_pii, _RE_CORREO.sub(_correo, texto)), conteos


def enmascarar_pii(texto: str) -> str:
    """Enmascara correos, tarjetas (Luhn) y teléfonos con marcadores estables."""
    return enmascarar_pii_con_conteos(texto)[0]
