"""Guardrail de entrada determinista (primera capa; sin LLM).

Detecta lo que los documentos mandan derivar a un humano: reembolsos por encima
del umbral (doc4), quejas sobre el trato de un empleado, disputas de facturación
y temas legales (doc5), además de intentos de manipulación del agente.

Decisiones:
- Reembolso: exige contexto de reembolso Y un monto estrictamente mayor al umbral.
  Un monto cuenta si lleva marcador de moneda ($, usd, dólares, pesos), en dígitos
  o en letras. Un número desnudo (sin moneda) solo cuenta si va justo tras una
  preposición que sigue a una palabra de reembolso ("reembolso de 800"): se
  prioriza no dejar pasar falsos negativos. Se ignoran ids como ORD-1200 y números
  de 7 o más dígitos sin separadores (teléfonos). Limitación conocida: un año
  tras "reembolso de" ("reembolso de 2024") se leería como monto.
- Formatos: "1,200" y "1.200" son 1200 (grupo de 3 dígitos); "1.200,50" y
  "1,200.50" usan el último separador como decimal; "500.01" es decimal.
- Precedencia (una sola categoría, el motivo menciona las demás):
  legal > manipulacion > facturacion > queja_trato > reembolso_alto.
- Fallo seguro: ante cualquier duda razonable se escala.
- Limitaciones conocidas (no se corrigen): "reembolso de 2024" se lee como monto;
  "$0.800" se interpreta como decimal (0.8); la palabra "legal" suelta escala por
  fallo seguro; las ventanas acotadas (60 caracteres) no ligan empleado y adjetivo
  si están más lejos.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tiendahogar_agent.config import Settings
from tiendahogar_agent.models import Accion
from tiendahogar_agent.texto import es_vacio_visible, quitar_tildes

CANAL_ESCALAMIENTO = "soporte@tiendahogar.example"

Categoria = Literal["reembolso_alto", "queja_trato", "facturacion", "legal", "manipulacion", "ninguna"]


class ResultadoGuardrail(BaseModel):
    """Veredicto del guardrail. La acción final la decide el orquestador salvo al escalar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    escalar: bool
    categoria: Categoria
    motivo: str
    regla: str | None = None
    accion: Accion | None = None
    canal: str | None = None


# ---------------- reglas por patrón (sobre texto sin tildes y en minúsculas) ----------------
def _re(*patrones: str) -> re.Pattern[str]:
    return re.compile("|".join(patrones))


_REGLAS_LEGAL = [
    ("legal_demanda", _re(
        r"\bdemand(?:ar|are|aria|aran|ado|aron|amos)\w*",
        r"\b(?:pon|interpon|present|inici)\w*\s+(?:una\s+)?demanda\b",
        r"\bdemanda\s+(?:legal|judicial|contra|formal)\b",
    )),
    ("legal_abogado", _re(r"\babogad[oa]s?\b", r"\bbufete\b")),
    ("legal_denuncia", _re(
        r"\bdenunci\w*",
    )),
    ("legal_tema", _re(
        r"\blegal(?:es|mente)?\b", r"\bilegal(?:es|mente)?\b", r"\btribunal(?:es)?\b",
        r"\bjuzgado\b", r"\bfiscalia\b", r"\bproteccion al consumidor\b",
        r"\bderechos del consumidor\b",
    )),
]

_REGLAS_MANIPULACION = [
    ("manip_ignorar_reglas", _re(
        r"\b(?:ignor|olvid|omit|salt|desobedec)\w*\s+(?:\w+\s+){0,3}"
        r"(?:instruccion|regla|restriccion|directriz|indicacion|prompt|politica)",
        r"\b(?:ignor|olvid)\w*\s+(?:todo\s+lo\s+anterior|lo\s+anterior|las\s+anteriores)",
        r"\b(?:ignore|disregard|forget|override)\s+(?:\w+\s+){0,3}(?:instructions|rules|prompt)",
        r"\bdisregard\s+the\s+(?:above|previous|prior)\b",
        r"\bforget\s+(?:everything|all)\b",
        r"\byou\s+are\s+now\b",
    )),
    ("manip_cambio_rol", _re(
        r"\bactua\s+como\b", r"\bcomportate\s+como\b", r"\b(?:finge|pretende|simula)\s+(?:ser|que\s+eres)\b",
        r"\bhaz\s+de\s+cuenta\s+que\s+eres\b", r"\bahora\s+eres\b", r"\bact\s+as\b",
    )),
    ("manip_revelar_prompt", _re(
        r"\b(?:revela|muestra|muestrame|dime|repite|imprime|ensename|comparte)\s+(?:\w+\s+){0,2}"
        r"(?:prompt|instrucciones|configuracion)",
        r"\bsystem\s+prompt\b", r"\bprompt\s+del\s+sistema\b", r"\bmodo\s+desarrollador\b",
        r"\bdeveloper\s+mode\b", r"\bjailbreak\b", r"\bsin\s+restricciones\b",
    )),
]

_REGLAS_FACTURACION = [
    ("facturacion_cobro", _re(
        r"\b(?:cobro|cargo)\s+(?:doble|duplicado|indebido|incorrecto|erroneo|extra|no\s+(?:reconocido|autorizado))\b",
        r"\bdoble\s+(?:cobro|cargo)\b",
        r"\bcobr\w*\s+(?:\w+\s+){0,2}(?:de\s+mas|dos\s+veces|doble|extra|mal)\b",
        r"\bcarg\w*\s+(?:\w+\s+){0,2}(?:de\s+mas|dos\s+veces|doble)\b",
        r"\bno\s+reconozco\s+(?:\w+\s+){0,2}(?:cobro|cargo)\b",
    )),
    ("facturacion_factura", _re(
        r"\bfactura\s+(?:(?:es|esta|estaba|salio|vino|viene)\s+)?(?:mal|incorrecta|erronea|equivocada|duplicada|con\s+error)\b",
        r"\bfactura\s+(?:(?:viene|vino|salio|esta|estaba)\s+)?con\s+(?:un\s+)?monto\s+(?:equivocado|incorrecto|erroneo)\b",
        r"\bme\s+facturaron\s+(?:mal|de\s+mas|doble|dos\s+veces)\b",
        r"\berror\s+en\s+(?:la|mi|esta)\s+factura\b",
        r"\bdisput\w*\s+(?:\w+\s+){0,2}(?:cobro|cargo|factura|facturacion)\b",
        r"\b(?:reclamo|problema|disputa)\s+(?:de|con)\s+(?:la\s+)?facturacion\b",
    )),
]

_EMPLEADO = (
    r"(?:empleado|empleada|vendedor|vendedora|agente|asesor|asesora|repartidor|repartidora|cajero|cajera|"
    r"trabajador|trabajadora|personal|dependiente|conductor|tecnico|tecnica|representante)"
)
_NEG = r"grosero|grosera|groseria|irrespetuos[oa]|descortes|insolente|hostil|maleducad[oa]|prepotente|altanero|altanera"
_NEGATIVO = rf"(?:{_NEG})"


class _ReglaVentana:
    """Regla «queja + trato/atención + rol» en cualquier orden, dentro de una misma oración.

    Exige los tres elementos con una separación máxima de `ventana` caracteres entre el primero y el
    último. Solo recorre las primeras apariciones de cada uno (coste acotado, sin backtracking).
    «Atención» solo cuenta junto a un rol (no captura «atención al cliente» como canal).
    """

    _MAX_POSICIONES = 8
    _ORACIONES = re.compile(r"[^.?!\n;]+")

    def __init__(self, *elementos: re.Pattern[str], ventana: int = 90) -> None:
        self._elementos = elementos
        self._ventana = ventana

    def search(self, texto: str) -> bool:
        for oracion in self._ORACIONES.finditer(texto):
            segmento = oracion.group()
            posiciones = []
            for patron in self._elementos:
                inicios = []
                for m in patron.finditer(segmento):
                    inicios.append(m.start())
                    if len(inicios) >= self._MAX_POSICIONES:
                        break
                if not inicios:
                    break
                posiciones.append(inicios)
            else:
                if _hay_ventana(posiciones, self._ventana):
                    return True
        return False


def _hay_ventana(posiciones: list[list[int]], ventana: int) -> bool:
    """¿Hay una elección (una posición por elemento) cuyo rango no supera la ventana?"""
    combos: list[tuple[int, ...]] = [()]
    for opciones in posiciones:
        combos = [c + (p,) for c in combos for p in opciones]
    return any(max(c) - min(c) <= ventana for c in combos)


_ROL = (
    r"(?:empleado|empleada|vendedor|vendedora|asesor|asesora|cajero|cajera|personal|gerente|encargado|"
    r"encargada|agente|mesero|mesera|dependiente|trabajador|trabajadora|funcionario|funcionaria|"
    r"representante|repartidor|repartidora|tecnico|tecnica|conductor|conductora|supervisor|supervisora)"
)
_QUEJAR = re.compile(r"\b(?:quej\w*|reclam\w*)\b")
_TRATO_O_ATENCION = re.compile(r"\b(?:trato|atencion)\b")
_ROL_RE = re.compile(rf"\b{_ROL}\b")
# «no tengo ninguna queja», «sin queja alguna»: se retira antes de evaluar la categoría.
_NEGACION_QUEJA = re.compile(
    r"\b(?:no\s+(?:tengo|tuve|hay|hubo|pongo|puse|presento)\s+(?:ninguna?\s+)?|ninguna?\s+|sin\s+)"
    r"(?:queja|reclamo)s?\b"
)

_REGLAS_TRATO = [
    ("trato_queja_rol", _ReglaVentana(_QUEJAR, _TRATO_O_ATENCION, _ROL_RE)),
    ("trato_maltrato", _re(
        r"\bmaltrat\w*", r"\bmal\s+trato\b", r"\bme\s+trat(?:o|aron)\s+mal\b",
        r"\btrato\s+(?:pesimo|horrible|inaceptable|grosero|indebido|humillante|malo)\b",
        r"\b(?:pesimo|horrible)\s+trato\b",
        r"\b(?:me|nos)\s+(?:insult|grit|humill|discrimin)\w*",
    )),
    ("trato_empleado", _re(
        rf"\b{_EMPLEADO}\b.{{0,60}}\b(?:{_NEG}|insult\w*|queja)\b",
        rf"\b{_NEGATIVO}\b.{{0,60}}\b{_EMPLEADO}\b",
        rf"\bqueja\w*\s+(?:\w+\s+){{0,3}}{_EMPLEADO}\b",
        rf"\b{_EMPLEADO}\b.{{0,60}}\bme\s+atendi(?:o|eron)\s+(?:muy\s+)?mal\b",
    )),
]

_CONTEXTO_REEMBOLSO = re.compile(
    r"reembols|reintegr|refund|me\s+(?:devuelv|regres)\w*|(?:devuelv|devolv|regres)\w*\s+(?:\w+\s+){0,2}dinero|"
    r"devolucion\s+(?:de|del)\s+(?:mi\s+)?dinero|mi\s+dinero\s+de\s+vuelta"
)

# ---------------- extracción de montos ----------------
_TOKEN = re.compile(r"\$|(?<![\d.,])\d{1,3}(?: \d{3})+(?!\d)|\d[\d.,]*\d|\d|[a-z]+")
_MONEDAS = {"dolar", "dolares", "usd", "peso", "pesos", "$"}
_PREPOSICIONES = {"de", "por", "valor", "monto", "total", "suma", "un", "una"}

_NUM_PALABRAS: dict[str, float] = {
    "medio": 0.5, "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100, "doscientos": 200,
    "trescientos": 300, "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600,
    "setecientos": 700, "ochocientos": 800, "novecientos": 900,
}
_MULTIPLICADORES = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}


def _parsear_numero(tok: str) -> float:
    """'1,200' -> 1200; '1 200' -> 1200; '1.200,50' -> 1200.5; '500.01' -> 500.01."""
    tok = tok.replace(" ", "")
    tiene_punto, tiene_coma = "." in tok, "," in tok
    if tiene_punto and tiene_coma:
        dec = "." if tok.rfind(".") > tok.rfind(",") else ","
        miles = "," if dec == "." else "."
        return float(tok.replace(miles, "").replace(dec, "."))
    if tiene_punto or tiene_coma:
        sep = "." if tiene_punto else ","
        partes = tok.split(sep)
        if len(partes) > 2 or (len(partes) == 2 and len(partes[1]) == 3):
            return float("".join(partes))
        return float(tok.replace(sep, "."))
    return float(tok)


def _valor_palabras(palabras: list[str]) -> float:
    total, actual = 0, 0
    for p in palabras:
        if p in _NUM_PALABRAS:
            actual += _NUM_PALABRAS[p]
        elif p in _MULTIPLICADORES:
            total += (actual or 1) * _MULTIPLICADORES[p]
            actual = 0
    return float(total + actual)


def _montos(texto: str) -> list[float]:
    """Montos monetarios (o desnudos tras 'reembolso de') hallados en el texto normalizado."""
    toks = _TOKEN.findall(texto)
    montos: list[float] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        inicio, largo = i, False
        if t[0].isdigit():
            valor = _parsear_numero(t)
            largo = t.isdigit() and len(t) >= 7
            i += 1
            if i < len(toks) and toks[i] in _MULTIPLICADORES:
                valor *= _MULTIPLICADORES[toks[i]]
                i += 1
        elif t in _NUM_PALABRAS or t in _MULTIPLICADORES:
            palabras: list[str] = []
            while i < len(toks) and (
                toks[i] in _NUM_PALABRAS
                or toks[i] in _MULTIPLICADORES
                or (toks[i] == "y" and palabras and i + 1 < len(toks) and toks[i + 1] in _NUM_PALABRAS)
            ):
                if toks[i] != "y":
                    palabras.append(toks[i])
                i += 1
            valor = _valor_palabras(palabras)
        else:
            i += 1
            continue
        previo = toks[inicio - 1] if inicio > 0 else ""
        siguiente = toks[i] if i < len(toks) else ""
        con_moneda = previo in ("$", "usd", "us") or siguiente in _MONEDAS
        antes = toks[max(0, inicio - 4):inicio]
        tras_reembolso = previo in _PREPOSICIONES and any(x.startswith(("reembols", "reintegr", "refund")) for x in antes)
        if previo == "ord" or (largo and not con_moneda):
            continue
        if con_moneda or tras_reembolso:
            montos.append(valor)
    return montos


def _hay_reembolso_alto(texto: str, umbral: float) -> bool:
    return bool(_CONTEXTO_REEMBOLSO.search(texto)) and any(m > umbral for m in _montos(texto))


# ---------------- evaluación ----------------
_PRIORIDAD: list[tuple[Categoria, str]] = [
    ("legal", "tema legal (doc5: derivar a un humano)"),
    ("manipulacion", "intento de manipular las instrucciones del agente"),
    ("facturacion", "disputa de facturación (doc5: derivar a un humano)"),
    ("queja_trato", "queja sobre el trato de un empleado (doc5: derivar a un humano)"),
    ("reembolso_alto", "reembolso por encima del umbral (doc4: requiere supervisor humano)"),
]
_REGLAS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "legal": _REGLAS_LEGAL,
    "manipulacion": _REGLAS_MANIPULACION,
    "facturacion": _REGLAS_FACTURACION,
    "queja_trato": _REGLAS_TRATO,
}


def evaluar_entrada(mensaje: str, umbral_reembolso: float) -> ResultadoGuardrail:
    """Evalúa el mensaje del cliente; nunca lanza ante entradas vacías o no textuales."""
    if not isinstance(mensaje, str) or es_vacio_visible(mensaje):
        return ResultadoGuardrail(escalar=False, categoria="ninguna", motivo="Sin contenido que evaluar.")
    texto = quitar_tildes(mensaje)
    disparadas: list[tuple[Categoria, str, str]] = []
    for categoria, descripcion in _PRIORIDAD:
        if categoria == "reembolso_alto":
            if _hay_reembolso_alto(texto, umbral_reembolso):
                disparadas.append((categoria, "reembolso_monto_alto", descripcion))
            continue
        base = _NEGACION_QUEJA.sub(" ", texto) if categoria == "queja_trato" else texto
        for regla, patron in _REGLAS[categoria]:
            if patron.search(base):
                disparadas.append((categoria, regla, descripcion))
                break
    if not disparadas:
        return ResultadoGuardrail(escalar=False, categoria="ninguna", motivo="Ninguna regla de escalamiento aplica.")
    categoria, regla, descripcion = disparadas[0]
    motivo = f"Se escala por {descripcion}."
    otras = [c for c, _, _ in disparadas[1:]]
    if otras:
        motivo += " También detectado: " + ", ".join(otras) + "."
    return ResultadoGuardrail(
        escalar=True, categoria=categoria, motivo=motivo, regla=regla,
        accion="escalar", canal=CANAL_ESCALAMIENTO,
    )


def evaluar_con_settings(mensaje: str, settings: Settings) -> ResultadoGuardrail:
    """Variante que toma el umbral de `Settings.umbral_reembolso`."""
    return evaluar_entrada(mensaje, settings.umbral_reembolso)
