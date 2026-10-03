"""Verificación de salida determinista (última capa de guardrails; sin LLM).

Revisa la respuesta candidata ANTES de entregarla. Si algo falla, se sustituye
por una respuesta segura y se escala a un humano (fallo seguro = escalar).

Reglas (identificadores estables en `reglas_fallidas`):
- `cifra_sin_sustento`: toda cifra de la respuesta (dígitos o letras) debe aparecer
  en los chunks recuperados, en el resultado de la tool o en el mensaje del usuario.
- `fuente_no_recuperada`: las fuentes citadas deben ser doc_id de los chunks recuperados; la
  fuente especial «pedidos» solo vale si `resultado_tool` trae un pedido consultado con éxito
  (ni un error de la tool ni la ausencia de llamada la sustentan).
- `compromiso_reembolso_aprobado`, `compromiso_garantia_resultado`,
  `compromiso_excepcion`: compromisos que el agente no puede asumir.
- `falta_canal_escalamiento`: si la acción es escalar, la respuesta nombra el canal.
- `respuesta_demasiado_larga`: la respuesta supera MAX_CARACTERES_RESPUESTA (20 000); se
  falla seguro sin analizarla (el contexto se recorta a 100 000 caracteres por texto).
- `respuesta_vacia`, `accion_invalida`: entradas inservibles (fallo seguro).

Cifras, decisiones:
- Se normaliza a valor numérico: «12» = «doce»; «1,200» = «1.200» = «1 200» = 1200;
  «2,5» = «2.5»; «5 mil» = «cinco mil»; «15 %» = «quince por ciento». Una cifra
  ambigua con un solo separador y tres decimales («1.200») admite lectura de miles
  y decimal; basta con que una lectura esté sustentada. «1 200» en el contexto
  también aporta 1 y 200 por separado (lectura lenient solo para el contexto).
- Coincidencia exacta del valor: NO se aceptan rangos ni aproximaciones («5-10 días»
  sustenta 5 y 10, no 7) ni conversiones de unidad («una semana» no equivale a 7 días).
- Números en letras: 0-29, decenas («treinta y cinco»), centenas, «mil», «millón/es»
  y combinaciones («dos mil quinientos», «ciento veinte»). Ordinales («segundo»),
  «doble», «mitad», «media», «cientos de», «miles de» NO son cifras.
- «un/una/uno» son artículos y NO cuentan, salvo que lo siguiente sea una unidad
  de tiempo o monto («un día», «una semana», «un peso») o «millón», o formen parte de
  un compuesto («treinta y un», «veintiún», «ciento un»). Limitación: «un día» como
  expresión vaga («algún día, un día cualquiera») se verifica como la cifra 1.
- IDs de pedido («ORD-1001», tolera espacios): se comparan completos contra
  tool/mensaje/contexto; el número 1001 suelto nunca se exige. Las cifras de los ids
  del contexto sí sustentan el número suelto (lenient). Fechas ISO (AAAA-MM-DD): se
  comparan completas; sus partes (año, mes, día) sustentan cifras sueltas del
  contexto («el 10 de octubre»).
- Los marcadores de lista al inicio de línea («1.», «2)») de la respuesta se ignoran.
- Limitaciones: no valida unidades ni el sentido de la cifra (solo que exista); los
  meses en letras no se convierten a número; «tres 5» o cifras pegadas a letras
  («2do») se leen como la cifra suelta.

Compromisos: la detección (léxica, de fallo seguro, con lista cerrada de negaciones y
de descripciones impersonales de política) vive en `guardrail_compromisos`; ahí está
documentada la frontera de decisión y sus limitaciones.

Verificar nunca lanza: entradas None o raras se tratan como vacías.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict

from tiendahogar_agent.guardrail_compromisos import (
    R_EXCEPCION,
    R_GARANTIA,
    R_NOTIFICACION,
    R_REEMBOLSO,
    detectar_compromisos,
)
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.models import Accion, Chunk
from tiendahogar_agent.texto import quitar_tildes

_log = logging.getLogger(__name__)

# Tope de longitud: más allá se falla seguro (acota el costo de la extracción de cifras).
MAX_CARACTERES_RESPUESTA = 20_000
# Cada texto de contexto se recorta a este tamaño antes de extraer cifras.
_MAX_CARACTERES_CONTEXTO = 100_000

RESPUESTA_SEGURA = (
    "Lo siento, no tengo esa información con certeza y prefiero no darte un dato "
    "incorrecto. Para que te ayuden bien, escríbele a nuestro equipo humano a "
    f"{CANAL_ESCALAMIENTO} y con gusto te atenderán."
)

_ACCIONES = ("responder", "escalar", "pedir_dato")

# Fuente que se cita al usar la tool de pedidos (no es un doc_id).
FUENTE_PEDIDOS = "pedidos"

R_CIFRA = "cifra_sin_sustento"
R_FUENTE = "fuente_no_recuperada"
R_CANAL = "falta_canal_escalamiento"
R_VACIA = "respuesta_vacia"
R_ACCION = "accion_invalida"
R_LARGA = "respuesta_demasiado_larga"


class ResultadoVerificacion(BaseModel):
    """Veredicto de la verificación; `respuesta`/`accion`/`canal` son los que se entregan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    respuesta: str
    accion: Accion
    canal: str | None = None
    fuentes: list[str]
    reglas_fallidas: tuple[str, ...] = ()
    detalles: tuple[str, ...] = ()


# ------------------------------------------------------------ extracción de cifras
_Valor = Fraction | str  # str solo para números gigantes (se comparan literalmente)

_ESCANEO = re.compile(
    r"(?P<fecha>(?<!\d)\d{4}-\d{2}-\d{2}(?!\d))"
    r"|(?P<pedido>\bord\s*-\s*\d+)"
    r"|(?P<agrupado>(?<![\d.,])\d{1,3}(?: \d{3})+(?!\d))"
    r"|(?P<num>\d+(?:[.,]\d+)*)"
    r"|(?P<palabra>[a-z]+)"
)
_POR_CIENTO = re.compile(r"\bpor\s+ciento\b")
_MARCADOR_LISTA = re.compile(r"(?m)^[ \t]*\d{1,2}[.)][ \t]+")

_UNIDADES_U = {"cero": 0, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
               "siete": 7, "ocho": 8, "nueve": 9}
_TEENS = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
          "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
          "veintiuno": 21, "veintiun": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23,
          "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
          "veintiocho": 28, "veintinueve": 29}
_DECENAS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
            "ochenta": 80, "noventa": 90}
_CENTENAS = {"cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300,
             "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600,
             "setecientos": 700, "ochocientos": 800, "novecientos": 900}
_CENTENAS.update({k[:-2] + "as": v for k, v in list(_CENTENAS.items()) if k.endswith("ientos")})
_VALOR: dict[str, int] = {**_UNIDADES_U, **_TEENS, **_DECENAS, **_CENTENAS}
_CATEGORIA: dict[str, str] = {
    **dict.fromkeys(_UNIDADES_U, "U"), **dict.fromkeys(_TEENS, "T"),
    **dict.fromkeys(_DECENAS, "D"), **dict.fromkeys(_CENTENAS, "C"),
    "uno": "U1", "un": "U1", "una": "U1", "mil": "M", "millon": "MM", "millones": "MM", "y": "Y",
}
_SIGUE: dict[str | None, set[str]] = {
    None: {"U", "U1", "T", "D", "C", "M", "MM"},
    "U": {"M", "MM"}, "U1": {"M", "MM"}, "T": {"M", "MM"},
    "D": {"Y", "M", "MM"},
    "C": {"D", "T", "U", "U1", "M", "MM"},
    "M": {"C", "D", "T", "U", "U1"},
    "MM": {"C", "D", "T", "U", "U1", "M", "MM"},
    "Y": {"U", "U1"},
}
_MULT = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}
_UNIDADES_MEDIDA = {
    "dia", "dias", "semana", "semanas", "mes", "meses", "ano", "anos", "hora", "horas",
    "minuto", "minutos", "unidad", "unidades", "peso", "pesos", "dolar", "dolares",
    "euro", "euros", "cuota", "cuotas",
}
_MAX_DIGITOS = 40


_GUIONES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")


def _norm(texto: str) -> str:
    """Minúsculas, sin tildes y con los guiones Unicode unificados a «-» (ids ORD)."""
    return quitar_tildes(texto).translate(_GUIONES)


def _val(texto: str) -> _Valor:
    """Fraction exacta; los números gigantes se guardan como texto (sin lanzar)."""
    if len(texto) > _MAX_DIGITOS:
        return texto.lstrip("0") or "0"
    return Fraction(texto)


def _figuras_de_token(tok: str) -> list[frozenset[_Valor]]:
    """Token numérico -> figuras; cada figura es el conjunto de lecturas posibles."""
    try:
        if tok.isdigit():
            return [frozenset({_val(tok)})]
        tiene_punto, tiene_coma = "." in tok, "," in tok
        if tiene_punto and tiene_coma:
            dec = "." if tok.rfind(".") > tok.rfind(",") else ","
            miles = "," if dec == "." else "."
            enteros, _, decimales = tok.rpartition(dec)
            grupos = enteros.split(miles)
            if miles not in decimales and len(grupos[0]) <= 3 and all(len(g) == 3 for g in grupos[1:]):
                return [frozenset({_val("".join(grupos) + "." + decimales)})]
            raise ValueError(tok)
        sep = "." if tiene_punto else ","
        partes = tok.split(sep)
        if len(partes) > 2:
            if len(partes[0]) <= 3 and all(len(p) == 3 for p in partes[1:]):
                return [frozenset({_val("".join(partes))})]
            raise ValueError(tok)
        ent, dec = partes
        decimal = _val(f"{ent}.{dec}")
        if len(dec) == 3 and len(ent) <= 3 and ent.strip("0"):
            return [frozenset({_val(ent + dec), decimal})]
        return [frozenset({decimal})]
    except ValueError:
        return [frozenset({_val(p)}) for p in re.split(r"\D+", tok) if p]


class _Cifras:
    """Cifras, ids y fechas extraídos de un texto."""

    def __init__(self) -> None:
        self.figuras: list[tuple[str, frozenset[_Valor]]] = []  # (literal, lecturas)
        self.ids: set[str] = set()
        self.fechas: set[str] = set()
        self.extra: set[_Valor] = set()  # cifras de apoyo (solo contexto)


def _contiguo(texto: str, a: tuple[str, str, int, int], b: tuple[str, str, int, int]) -> bool:
    return not texto[a[3]:b[2]].strip()


def _leer_numero(toks: list[tuple[str, str, int, int]], i: int, texto: str) -> tuple[int, int] | None:
    """Número en letras desde toks[i]: devuelve (valor, índice siguiente) o None."""
    n = len(toks)
    j, prev = i, None
    total = actual = 0
    while j < n and toks[j][0] == "palabra":
        if j > i and not _contiguo(texto, toks[j - 1], toks[j]):
            break
        w = toks[j][1]
        cat = _CATEGORIA.get(w)
        if cat is None or cat not in _SIGUE[prev]:
            break
        sig = toks[j + 1] if j + 1 < n and toks[j + 1][0] == "palabra" else None
        sig_ok = sig is not None and _contiguo(texto, toks[j], sig)
        if cat == "Y" and not (sig_ok and sig is not None and _CATEGORIA.get(sig[1]) in ("U", "U1")):
            break
        if cat == "U1" and prev is None and not (
            sig_ok and sig is not None and (sig[1] in _UNIDADES_MEDIDA or _CATEGORIA.get(sig[1]) == "MM")
        ):
            break  # artículo «un/una/uno»
        if cat in ("U", "T", "D", "C"):
            actual += _VALOR[w]
        elif cat == "U1":
            actual += 1
        elif cat in ("M", "MM"):
            total += (actual or 1) * _MULT[w]
            actual = 0
        prev = cat
        j += 1
    if j == i or prev == "Y":
        return None
    return total + actual, j


def _extraer(texto: str, *, contexto: bool) -> _Cifras:
    """Extrae cifras/ids/fechas de un texto ya normalizado (sin tildes, minúsculas)."""
    texto = _POR_CIENTO.sub("porciento", texto)
    toks = [(m.lastgroup or "", m.group(), m.start(), m.end()) for m in _ESCANEO.finditer(texto)]
    out = _Cifras()
    i, n = 0, len(toks)
    while i < n:
        tipo, val, _ini, _fin = toks[i]
        if tipo == "fecha":
            out.fechas.add(val)
            if contexto:
                out.extra.update(_val(p) for p in val.split("-"))
            i += 1
        elif tipo == "pedido":
            out.ids.add(re.sub(r"\s", "", val))
            if contexto:
                out.extra.add(_val(re.sub(r"\D", "", val)))
            i += 1
        elif tipo in ("agrupado", "num"):
            if tipo == "agrupado":
                figuras = [frozenset({_val(val.replace(" ", ""))})]
                if contexto:
                    out.extra.update(_val(p) for p in val.split(" "))
            else:
                figuras = _figuras_de_token(val)
            literal = val
            i += 1
            if i < n and toks[i][0] == "palabra" and toks[i][1] in _MULT and _contiguo(texto, toks[i - 1], toks[i]):
                m = _MULT[toks[i][1]]
                figuras = [frozenset(v * m if isinstance(v, Fraction) else v for v in f) for f in figuras]
                literal += " " + toks[i][1]
                i += 1
            out.figuras.extend((literal, f) for f in figuras)
        else:
            leido = _leer_numero(toks, i, texto)
            if leido is None:
                i += 1
                continue
            valor, j = leido
            out.figuras.append((texto[toks[i][2]:toks[j - 1][3]], frozenset({Fraction(valor)})))
            i = j
    return out


def _aplanar(valor: Any, partes: list[str], profundidad: int = 0) -> None:
    """Recoge los valores (no las claves) de un resultado de tool de forma segura."""
    if valor is None or isinstance(valor, bool) or profundidad > 6:
        return
    if isinstance(valor, dict):
        for v in valor.values():
            _aplanar(v, partes, profundidad + 1)
    elif isinstance(valor, (list, tuple, set, frozenset)):
        for v in valor:
            _aplanar(v, partes, profundidad + 1)
    else:
        partes.append(str(valor))


def _hay_pedido_valido(resultado_tool: Any) -> bool:
    """True si hubo una consulta de pedido exitosa (sin clave `error`).

    Acepta el dict de un pedido o `{"pedidos": [dict, ...]}` (varias consultas en el turno).
    """
    if not isinstance(resultado_tool, dict) or not resultado_tool:
        return False
    if isinstance(resultado_tool.get("pedidos"), list):
        return any(isinstance(p, dict) and p and "error" not in p for p in resultado_tool["pedidos"])
    return "error" not in resultado_tool


def _texto_de(valor: Any) -> str:
    return valor if isinstance(valor, str) else ""


def _verificar_cifras(respuesta: str, contexto: Iterable[str]) -> list[str]:
    """Devuelve detalles de cifras/ids/fechas de la respuesta sin sustento en el contexto."""
    ctx = _Cifras()
    for texto in contexto:
        c = _extraer(_norm(texto[:_MAX_CARACTERES_CONTEXTO]), contexto=True)
        ctx.ids |= c.ids
        ctx.fechas |= c.fechas
        ctx.extra |= c.extra
        for _lit, lecturas in c.figuras:
            ctx.extra |= lecturas
    resp = _extraer(_norm(_MARCADOR_LISTA.sub("", respuesta)), contexto=False)
    detalles: list[str] = []
    vistos: set[str] = set()

    def marca(etiqueta: str, literal: str) -> None:
        d = f"{etiqueta} sin sustento en el contexto: «{literal}»"
        if d not in vistos:
            vistos.add(d)
            detalles.append(d)

    for lit in sorted(resp.ids - ctx.ids):
        marca("id de pedido", lit)
    for lit in sorted(resp.fechas - ctx.fechas):
        marca("fecha", lit)
    for literal, lecturas in resp.figuras:
        if not (lecturas & ctx.extra):
            marca("cifra", literal)
    return detalles


# ---------------------------------------------------------------------- API
def _fallo(reglas: list[str], detalles: list[str]) -> ResultadoVerificacion:
    return ResultadoVerificacion(
        ok=False, respuesta=RESPUESTA_SEGURA, accion="escalar", canal=CANAL_ESCALAMIENTO,
        fuentes=[], reglas_fallidas=tuple(reglas), detalles=tuple(detalles),
    )


def verificar_salida(
    respuesta: str,
    accion: Accion,
    fuentes: list[str],
    chunks: list[Chunk],
    resultado_tool: dict | None,
    mensaje_usuario: str,
) -> ResultadoVerificacion:
    """Verifica la respuesta candidata. Nunca lanza; ante cualquier duda, escala."""
    try:
        return _verificar(respuesta, accion, fuentes, chunks, resultado_tool, mensaje_usuario)
    except Exception as exc:  # noqa: BLE001  defensa final: fallo seguro = escalar
        _log.error("fallo inesperado al verificar la salida: %s", type(exc).__name__)
        return _fallo(["error_verificacion"], [f"error interno al verificar: {type(exc).__name__}"])


def _verificar(
    respuesta: str,
    accion: Accion,
    fuentes: list[str],
    chunks: list[Chunk],
    resultado_tool: dict | None,
    mensaje_usuario: str,
) -> ResultadoVerificacion:
    texto = _texto_de(respuesta)
    reglas: list[str] = []
    detalles: list[str] = []

    if accion not in _ACCIONES:
        reglas.append(R_ACCION)
        detalles.append(f"acción no reconocida: {accion!r}")
    if not texto.strip():
        reglas.append(R_VACIA)
        detalles.append("la respuesta está vacía")

    if len(texto) > MAX_CARACTERES_RESPUESTA:
        return _fallo(
            reglas + [R_LARGA],
            detalles + [f"la respuesta supera {MAX_CARACTERES_RESPUESTA} caracteres"],
        )

    lista_chunks = [c for c in (chunks if isinstance(chunks, list) else []) if isinstance(c, Chunk)]
    fuentes_ok = [f for f in fuentes if isinstance(f, str)] if isinstance(fuentes, list) else []

    partes_tool: list[str] = []
    _aplanar(resultado_tool, partes_tool)
    contexto = [c.texto for c in lista_chunks] + partes_tool + [_texto_de(mensaje_usuario)]
    det_cifras = _verificar_cifras(texto, contexto)
    if det_cifras:
        reglas.append(R_CIFRA)
        detalles.extend(det_cifras)

    recuperados = {c.doc_id for c in lista_chunks}
    if _hay_pedido_valido(resultado_tool):
        recuperados.add(FUENTE_PEDIDOS)
    ajenas = [f for f in dict.fromkeys(fuentes_ok) if f not in recuperados]
    if ajenas:
        reglas.append(R_FUENTE)
        detalles.append("fuentes no recuperadas: " + ", ".join(ajenas))

    por_regla: dict[str, str] = {}
    for regla, frase in detectar_compromisos(texto):
        por_regla.setdefault(regla, frase)
    for regla in (R_REEMBOLSO, R_GARANTIA, R_EXCEPCION, R_NOTIFICACION):
        if regla in por_regla:
            reglas.append(regla)
            detalles.append(f"compromiso no autorizado: «{por_regla[regla]}»")

    if accion == "escalar" and CANAL_ESCALAMIENTO not in quitar_tildes(texto):
        reglas.append(R_CANAL)
        detalles.append(f"la respuesta no incluye el canal {CANAL_ESCALAMIENTO}")

    if reglas:
        return _fallo(reglas, detalles)
    return ResultadoVerificacion(
        ok=True, respuesta=respuesta, accion=accion,
        canal=CANAL_ESCALAMIENTO if accion == "escalar" else None,
        fuentes=fuentes_ok,
    )
