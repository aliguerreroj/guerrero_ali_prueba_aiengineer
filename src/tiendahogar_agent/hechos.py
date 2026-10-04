"""Tabla de hechos críticos y detector de contradicciones (ADR-009, regla R_HECHO de T10).

Un «hecho» es un dato verificable de los documentos que el agente suele afirmar: la
garantía por categoría de producto (doc1) y el plazo de envío por destino (doc3). La tabla
es CERRADA y se deriva a mano de los documentos; cada hecho guarda la `frase_origen`
literal (un test comprueba que existe tal cual en `data/docs/`).

Detección (léxica y determinista, sin LLM; no es comprensión semántica):
- El texto se divide en oraciones (`. ? ! ;` seguidos de espacio, o salto de línea) y se
  normaliza sin tildes ni mayúsculas.
- Cifra con unidad: «12 meses», «doce meses», «5-7 días hábiles», «5 a 7 días hábiles»,
  «entre 5 y 7 días hábiles», «un mes». Una cifra solo se compara si aparece junto a la PALABRA
  CLAVE de su hecho y con su UNIDAD (acotación tras el eval real T19, decisión del humano):
  garantía = palabra «garantía» + «meses» («garantía de 12 meses», «12 meses de garantía»);
  envío = palabra de envío (envío, entrega, llega, despacho, demora, tarda, recibe) + «días
  hábiles» («el envío a la capital tarda 2-3 días hábiles»). «30 días para devolverlo», «hace
  2 meses» o «2 meses de uso» no activan nada.
- «Junto a»: la palabra clave está a lo sumo a 12 palabras (separadas por espacio) antes o
  después de la cifra, en la misma oración y sin cruzar un corte fuerte (coma, paréntesis,
  «pero», «sino», «aunque»). 12 es el mínimo que conserva los casos coordinados del conjunto
  de pruebas («llega a otras ciudades en 5-7 días hábiles y a la capital en 2-3 días hábiles»).
- Cada cifra se asocia al término más cercano que la precede (del mismo tipo); si ninguno
  la precede, al más cercano que la sigue. Así «la licuadora tiene 6 meses y la
  refrigeradora 12» no falla y «la licuadora tiene 12 meses» sí.
- Una cifra suelta es correcta si está dentro del rango del hecho; un rango citado debe ser
  igual al del hecho. Cifras sin unidad no se evalúan.
- Negación: si en las 6 palabras previas de la misma cláusula (se corta en coma, dos
  puntos, «pero», «sino», «y», «aunque») aparece «no/ni/nunca/tampoco», la cifra se
  considera negada y no se evalúa («la licuadora no tiene 12 meses, sino 6»). Puede dejar
  pasar un error muy mal redactado; es un límite documentado. También se ignoran las
  cifras precedidas de «hace» («la compraste hace 12 meses»).
- Productos NO listados (microondas, televisor...): ninguna garantía con cifra en meses
  puede asignárseles; se reporta con `hecho=None`.

Límite aceptado por decisión del humano: una redacción sin la palabra «garantía» o sin «días
hábiles» («el envío tarda 10 días», «la licuadora dura 12 meses», «5-7 días» a secas) YA NO se
verifica; se prefiere dejar pasar eso a bloquear respuestas correctas que mezclan cifras de
otros hechos (plazo de devolución de 30 días, antigüedad de la compra, garantía).

Límites: heurística léxica con lista cerrada de productos y destinos; no convierte unidades
(«un año» no es «12 meses»; «una semana» no es «7 días»); no sabe que «nevera» es una
refrigeradora (solo se reconocen los términos de la tabla, más variantes morfológicas
singular/plural/masculino de los mismos); «demás ciudades» se admite como variante de
«otras ciudades» solo para detectar.

Grupos y emparejamiento (ADR-009): términos separados solo por conectores («y», «así como»,
«la misma que la de», «tiene lo mismo que»...) forman un grupo coordinado. Una cifra (o rango)
se evalúa contra TODOS los términos del grupo. Cifras coordinadas («6 y 12 meses», «6 meses y
12 meses») se emparejan en orden con los términos del grupo si hay el mismo número de cada
uno; si no, no se evalúan (lado seguro: no bloquear respuestas posiblemente correctas). Con
varias corridas de cifras separadas («tiene 6 meses y la lavadora 12 meses»), la primera pieza
de la oración fija la dirección (cifra primero: miran a los términos que las siguen; término
primero: a los que las preceden) y ningún grupo cruza otra corrida. Ya no se admite el falso
positivo de «la licuadora y la lavadora tienen 6 y 12 meses».
Cifra atribuida («dices que», «me dijeron», «según tú», «mencionas»...) en su cláusula: no se
evalúa. Solo se examinan los 100 caracteres previos a cada cifra (coste lineal).
Límite: un rango sin unidad compartido («2-3 y 5-7 días») no se reconoce como dos cifras.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction

from tiendahogar_agent.texto import quitar_tildes

GARANTIA = "garantia"
ENVIO = "envio"

_GUIONES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")


@dataclass(frozen=True)
class Hecho:
    """Dato verificable de un documento.

    `terminos`: sinónimos/plurales/singulares ya normalizados (minúsculas, sin tildes).
    `minimo`/`maximo`: valor admitido (garantía: ambos iguales; envío: rango en días hábiles).
    `unidad`: «meses» o «días hábiles». `frase_origen`: oración literal de `data/docs/`.
    """

    clave: str
    tipo: str
    terminos: tuple[str, ...]
    minimo: int
    maximo: int
    unidad: str
    doc_id: str
    frase_origen: str

    @property
    def valor_texto(self) -> str:
        cifra = str(self.minimo) if self.minimo == self.maximo else f"{self.minimo}-{self.maximo}"
        return f"{cifra} {self.unidad}"

    def descripcion(self) -> str:
        """Formato legible: «licuadora: 6 meses (doc1)»."""
        return f"{self.clave}: {self.valor_texto} ({self.doc_id})"


_F_GRANDES = (
    "Todos los electrodomésticos grandes (refrigeradoras, lavadoras, estufas) tienen "
    "garantía de 12 meses desde la fecha de compra."
)
_F_PEQUENOS = (
    "Electrodomésticos pequeños (licuadoras, planchas, tostadoras) tienen garantía de 6 meses."
)
_F_CAPITAL = "Envíos a la capital: 2-3 días hábiles."
_F_OTRAS = "Envíos a otras ciudades: 5-7 días hábiles."


def _garantia(clave: str, terminos: tuple[str, ...], meses: int, frase: str) -> Hecho:
    return Hecho(clave, GARANTIA, terminos, meses, meses, "meses", "doc1", frase)


HECHOS: tuple[Hecho, ...] = (
    _garantia("refrigeradora", ("refrigeradoras", "refrigeradora", "refrigeradores", "refrigerador"), 12, _F_GRANDES),
    _garantia("lavadora", ("lavadoras", "lavadora"), 12, _F_GRANDES),
    _garantia("estufa", ("estufas", "estufa"), 12, _F_GRANDES),
    _garantia("licuadora", ("licuadoras", "licuadora"), 6, _F_PEQUENOS),
    _garantia("plancha", ("planchas", "plancha"), 6, _F_PEQUENOS),
    _garantia("tostadora", ("tostadoras", "tostadora"), 6, _F_PEQUENOS),
    Hecho("envío a la capital", ENVIO, ("capital",), 2, 3, "días hábiles", "doc3", _F_CAPITAL),
    Hecho(
        "envío a otras ciudades", ENVIO,
        ("fuera de la capital", "otras ciudades", "otra ciudad", "demas ciudades"),
        5, 7, "días hábiles", "doc3", _F_OTRAS,
    ),
)

# Productos conocidos que NO figuran en doc1: no se les puede asignar una garantía en meses.
NO_LISTADOS: tuple[str, ...] = (
    "microondas", "microonda", "televisores", "televisor", "television", "tv",
    "aires acondicionados", "aire acondicionado", "ventiladores", "ventilador",
    "aspiradoras", "aspiradora", "cafeteras", "cafetera", "secadoras", "secadora",
    "lavavajillas", "batidoras", "batidora", "freidoras", "freidora", "arroceras", "arrocera",
    "calentadores", "calentador", "congeladores", "congelador",
)

_UNI = ("uno", "un", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve")
_VALOR_UNI = {"uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
              "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
_VALOR_TEEN = {"cero": 0, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
               "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
               "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21,
               "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
               "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
               "veintinueve": 29}
_VALOR_DEC = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
              "ochenta": 80, "noventa": 90}


def _alt(palabras: list[str] | tuple[str, ...]) -> str:
    return "|".join(sorted(map(re.escape, palabras), key=len, reverse=True))


_NUM_PALABRA = (
    rf"(?:(?:{_alt(list(_VALOR_DEC))})(?:\s+y\s+(?:{_alt(_UNI)})\b)?\b"
    rf"|(?:{_alt(list(_VALOR_TEEN) + list(_UNI))})\b)"
)
_NUM = rf"(?:\d+(?:[.,]\d+)?|{_NUM_PALABRA})"
_CIFRA = re.compile(
    rf"(?<![\w.,])(?P<a>{_NUM})"
    rf"(?:\s*(?P<sep>-|\ba\b|\bal\b|\by\b|\bo\b|\bhasta\b)\s*(?P<b>{_NUM}))?"
    r"\s*(?P<u>meses|mes|dias(?:\s+habiles)?|dia(?:\s+habil)?)\b"
)
_HACE = re.compile(r"\bhace(?:\s+(?:mas|menos)\s+de|\s+casi|\s+unos)?\s*$")
_CLAUSULA = re.compile(r"[,:(]|\b(?:pero|sino|y|aunque)\b")
_NEGADORES = frozenset({"no", "ni", "nunca", "tampoco"})
_VENTANA_NEGACION = 6
# Palabras que pueden separar dos términos coordinados (grupo: una cifra para todos).
_CONECTORES = frozenset({
    "y", "e", "o", "u", "ni", "como", "tanto", "que", "la", "las", "el", "los", "lo", "de", "del",
    "al", "a", "es", "son", "misma", "mismo", "mismas", "mismos", "igual", "iguales", "tu", "tus",
    "su", "sus", "mi", "mis", "un", "una", "unos", "unas", "tambien", "ademas", "mas", "con",
    "para", "en", "garantia", "asi", "tiene", "tienen", "respectivamente",
})
_MAX_HUECO = 8
# Cifra atribuida a otra persona («dices que...», «me dijeron...»): no se evalúa en esa cláusula.
_ATRIBUIDA = re.compile(
    r"\b(?:dices|dijiste|dice|dijeron|mencionas|mencionaste|comentas|comentaste|escuchaste|"
    r"leiste|afirmas|segun\s+tu)\b"
)
_ENTRE = re.compile(r"\bentre\s*$")
# Solo se examinan los últimos caracteres previos a cada cifra (coste lineal, no cuadrático).
_VENTANA_CARACTERES = 100
# Palabra clave de cada hecho: la cifra solo se compara si la tiene «junto a» (ADR-009).
_CUE_ENVIO = re.compile(r"\b(?:envi|entreg|lleg|despach|demor|tard|recib)")
_CUE_GARANTIA = re.compile(r"\bgarantia")
# «Junto a»: palabra clave a lo sumo a 12 palabras (separadas por espacio) antes o después de la
# cifra, en la misma oración y sin cruzar un corte fuerte (coma, paréntesis, «pero», «sino»,
# «aunque»). 12 es el mínimo que conserva los casos coordinados del conjunto de pruebas
# («llega a otras ciudades en 5-7 días hábiles y a la capital en 2-3 días hábiles»).
_VENTANA_PALABRAS = 12
_CORTE_FUERTE = re.compile(r"[,(]|\b(?:pero|sino|aunque)\b")
_SEPARADOR_ORACION = re.compile(r"(?<=[.?!;])\s+|\n+")


def _patron_terminos(terminos: set[str] | tuple[str, ...] | list[str]) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w)(?:{_alt(list(terminos))})(?!\w)")


_TERMINOS_GARANTIA: dict[str, Hecho | None] = {}
_TERMINOS_ENVIO: dict[str, Hecho | None] = {}
for _h in HECHOS:
    _destino = _TERMINOS_GARANTIA if _h.tipo == GARANTIA else _TERMINOS_ENVIO
    for _t in _h.terminos:
        _destino[_t] = _h
for _t in NO_LISTADOS:
    _TERMINOS_GARANTIA[_t] = None
_RE_GARANTIA = _patron_terminos(list(_TERMINOS_GARANTIA))
_RE_ENVIO = _patron_terminos(list(_TERMINOS_ENVIO))


@dataclass(frozen=True)
class Discrepancia:
    """Cifra de la respuesta que contradice la tabla (`hecho=None`: producto no listado)."""

    hecho: Hecho | None
    termino: str
    cifra: str

    def detalle(self) -> str:
        """Texto legible para el reintento: hecho correcto, documento y frase de origen."""
        if self.hecho is not None:
            return (
                f"«{self.termino}» con «{self.cifra}» es incorrecto; correcto: "
                f"{self.hecho.descripcion()}. Fuente ({self.hecho.doc_id}): "
                f"«{self.hecho.frase_origen}»"
            )
        return (
            f"«{self.termino}» no figura en la política de garantía (doc1) y no se le puede "
            f"asignar «{self.cifra}». Fuente (doc1): «{_F_GRANDES}» «{_F_PEQUENOS}» "
            "Di que no aparece listado, sin suponer ni dar un plazo."
        )


def _valor(texto: str) -> Fraction:
    if texto[0].isdigit():
        return Fraction(texto.replace(",", "."))
    partes = texto.split()
    total = _VALOR_DEC.get(partes[0], 0) or _VALOR_TEEN.get(partes[0], 0) or _VALOR_UNI.get(partes[0], 0)
    if len(partes) == 3:
        total += _VALOR_UNI[partes[2]]
    return Fraction(total)


def _clausula_previa(prefijo: str) -> str:
    """Texto de la cláusula en curso (desde la última coma, «:», «pero», «sino», «y»...)."""
    corte = 0
    for m in _CLAUSULA.finditer(prefijo):
        corte = m.end()
    return prefijo[corte:]


def _omitida(oracion: str, c: re.Match[str]) -> bool:
    """Cifra que no se evalúa: «hace N meses», negada o atribuida a otra persona."""
    prefijo = oracion[max(0, c.start() - _VENTANA_CARACTERES):c.start()]
    if _HACE.search(prefijo):
        return True
    clausula = _clausula_previa(prefijo[-80:])
    palabras = re.findall(r"\w+", clausula)[-_VENTANA_NEGACION:]
    if any(p in _NEGADORES for p in palabras):
        return True
    return bool(_ATRIBUIDA.search(_clausula_previa(prefijo)))


def _solo_conectores(oracion: str, ini: int, fin: int) -> bool:
    hueco = re.findall(r"\w+", oracion[ini:fin])
    return len(hueco) <= _MAX_HUECO and all(p in _CONECTORES for p in hueco)


def _grupo(
    oracion: str, terminos: list[re.Match[str]], i: int, *, hacia_atras: bool, lo: int, hi: int
) -> list[int]:
    """Índices (ascendentes) de los términos coordinados con `terminos[i]` dentro de [lo, hi].

    «Las licuadoras y las refrigeradoras tienen 12 meses»: entre términos coordinados solo hay
    conectores. Se extiende hacia atrás si la cifra va después del término, y hacia delante si
    lo precede.
    """
    grupo = [i]
    paso = -1 if hacia_atras else 1
    j = i + paso
    while lo <= j <= hi:
        a, b = (terminos[j], terminos[j - paso]) if hacia_atras else (terminos[j - paso], terminos[j])
        if not _solo_conectores(oracion, a.end(), b.start()):
            break
        grupo.append(j)
        j += paso
    return sorted(grupo)


def _compatible(hecho: Hecho, a: Fraction, b: Fraction | None) -> bool:
    if b is None:
        return hecho.minimo <= a <= hecho.maximo
    return (a, b) == (hecho.minimo, hecho.maximo)


_Item = tuple[Fraction, Fraction | None, str]  # (cifra, extremo del rango o None, literal)


def _item(c: re.Match[str]) -> _Item:
    a = _valor(c.group("a"))
    b = _valor(c.group("b")) if c.group("b") else None
    if b is not None and b < a:
        a, b = b, a
    return a, b, c.group().strip()


def _agrupar_cifras(oracion: str, cifras: list[re.Match[str]]) -> list[list[re.Match[str]]]:
    """Cifras consecutivas separadas solo por conectores («6 meses y 12 meses») van juntas."""
    corridas: list[list[re.Match[str]]] = []
    for c in cifras:
        if corridas and _solo_conectores(oracion, corridas[-1][-1].end(), c.start()):
            corridas[-1].append(c)
        else:
            corridas.append([c])
    return corridas


def _evaluar(
    oracion: str, terminos: list[re.Match[str]], cifras: list[re.Match[str]],
    tabla: dict[str, Hecho | None], salida: list[Discrepancia], tope: int,
) -> None:
    """Asocia cada corrida de cifras con su grupo de términos y reporta las contradicciones.

    - Una cifra (o rango) frente a un grupo coordinado de n términos: se evalúa contra todos.
    - n cifras coordinadas («6 y 12 meses», «6 meses y 12 meses») frente a n términos
      coordinados: se emparejan en orden. Si no coinciden en número, no se evalúan (lado seguro:
      no bloquear respuestas posiblemente correctas).
    - Con varias corridas, la dirección la fija el primer elemento de la oración: si es una cifra,
      cada corrida mira a los términos que la SIGUEN; si es un término, a los que la PRECEDEN. Los
      términos de una corrida nunca cruzan otra corrida.
    """
    pos = [m.start() for m in terminos]
    corridas = _agrupar_cifras(oracion, cifras)
    hacia_delante = cifras[0].start() < terminos[0].start()
    for k, corrida in enumerate(corridas):
        if len(salida) >= tope:
            return
        if hacia_delante:
            lo = bisect_left(pos, corrida[-1].end())
            hi = (bisect_left(pos, corridas[k + 1][0].start()) if k + 1 < len(corridas)
                  else len(terminos)) - 1
            if lo > hi:
                continue
            ancla = lo
        else:
            lo = bisect_left(pos, corridas[k - 1][-1].end()) if k else 0
            hi = bisect_right(pos, corrida[0].start()) - 1
            if lo > hi:
                continue
            ancla = hi
        grupo = _grupo(oracion, terminos, ancla, hacia_atras=not hacia_delante, lo=lo, hi=hi)
        items = [_item(c) for c in corrida]
        c0 = corrida[0]
        if (
            len(corrida) == 1 and c0.group("sep") in ("y", "o") and len(grupo) >= 2
            and not _ENTRE.search(oracion[max(0, c0.start() - 10):c0.start()])
        ):
            # «6 y 12 meses» con dos términos coordinados son dos cifras, no un rango.
            items = [(_valor(c0.group("a")), None, c0.group().strip()),
                     (_valor(c0.group("b")), None, c0.group().strip())]
        if len(items) == 1:
            pares = [(j, items[0]) for j in grupo]
        elif len(items) == len(grupo):
            pares = list(zip(grupo, items, strict=True))
        else:
            continue
        for j, (a, b, literal) in pares:
            termino = terminos[j].group()
            hecho = tabla[termino]
            if hecho is None:  # producto no listado: ninguna garantía con cifra es válida
                salida.append(Discrepancia(None, termino, literal))
            elif not _compatible(hecho, a, b):
                salida.append(Discrepancia(hecho, termino, literal))
            if len(salida) >= tope:
                return


class _Contexto:
    """Índices de una oración para decidir en O(log n) si una cifra tiene su palabra clave cerca."""

    def __init__(self, oracion: str) -> None:
        self.n = len(oracion)
        self.palabras = [m.start() for m in re.finditer(r"\S+", oracion)]
        self.cortes = [(m.start(), m.end()) for m in _CORTE_FUERTE.finditer(oracion)]
        self.corte_ini = [c[0] for c in self.cortes]
        self.corte_fin = [c[1] for c in self.cortes]
        self._claves: dict[re.Pattern[str], list[int]] = {}
        self._oracion = oracion

    def _posiciones(self, clave: re.Pattern[str]) -> list[int]:
        if clave not in self._claves:
            self._claves[clave] = [m.start() for m in clave.finditer(self._oracion)]
        return self._claves[clave]

    def cerca(self, c: re.Match[str], clave: re.Pattern[str]) -> bool:
        """¿Hay `clave` a <= 12 palabras de la cifra, sin cruzar un corte fuerte?"""
        pos = self._posiciones(clave)
        if not pos:
            return False
        k = bisect_right(self.palabras, c.start()) - 1
        desde = self.palabras[max(0, k - _VENTANA_PALABRAS)]
        i = bisect_right(self.corte_ini, c.start()) - 1  # último corte que empieza antes
        if i >= 0:
            desde = max(desde, self.corte_fin[i])
        j = bisect_left(pos, desde)
        if j < len(pos) and pos[j] < c.start():
            return True
        k = bisect_left(self.palabras, c.end()) + _VENTANA_PALABRAS
        hasta = self.palabras[k] if k < len(self.palabras) else self.n
        i = bisect_left(self.corte_ini, c.end())
        if i < len(self.cortes):
            hasta = min(hasta, self.corte_ini[i])
        j = bisect_left(pos, c.end())
        return j < len(pos) and pos[j] < hasta


def _revisar_oracion(oracion: str, salida: list[Discrepancia], tope: int) -> None:
    todas = list(_CIFRA.finditer(oracion))
    if not todas:
        return
    gar = list(_RE_GARANTIA.finditer(oracion))
    env = list(_RE_ENVIO.finditer(oracion))
    # (términos, tabla, unidad que exige la cifra, palabra clave que debe acompañarla)
    reglas = (
        (gar, _TERMINOS_GARANTIA, "mes", _CUE_GARANTIA),
        (env, _TERMINOS_ENVIO, "dias habiles", _CUE_ENVIO),
    )
    contexto: _Contexto | None = None
    for terminos, tabla, unidad, clave in reglas:
        if not terminos:
            continue
        contexto = contexto or _Contexto(oracion)
        cifras = [
            c for c in todas
            if re.sub(r"\s+", " ", c.group("u")).startswith(unidad)
            and contexto.cerca(c, clave) and not _omitida(oracion, c)
        ]
        if cifras:
            _evaluar(oracion, terminos, cifras, tabla, salida, tope)


def detectar_discrepancias(texto: str, maximo: int = 10) -> list[Discrepancia]:
    """Cifras de `texto` que contradicen la tabla de hechos (lista vacía si todo coincide)."""
    if not isinstance(texto, str) or not texto:
        return []
    normal = quitar_tildes(texto).translate(_GUIONES)
    salida: list[Discrepancia] = []
    for oracion in _SEPARADOR_ORACION.split(normal):
        if oracion.strip():
            _revisar_oracion(oracion, salida, maximo)
        if len(salida) >= maximo:
            break
    return salida[:maximo]
