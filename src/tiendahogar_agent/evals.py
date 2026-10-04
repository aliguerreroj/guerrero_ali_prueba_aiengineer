"""Runner de evals (T19): ejecuta `evals/golden_set.json` y mide al agente por categoría.

Uso (con el entorno activado):

    python -m tiendahogar_agent.evals [--golden RUTA] [--max-costo USD] [--salida CARPETA]

o, equivalente, `python harness/init.py --full`. El LLM real (Anthropic) SOLO se usa si hay
`ANTHROPIC_API_KEY` (la lee `Settings`, que carga el `.env`); sin clave se omite con un aviso y
sale con código 0. Un tope de costo (`--max-costo`, 2 USD por defecto) detiene la ejecución de
forma limpia, guarda el reporte parcial y sale con código 3.

Qué mide, por categoría y global: acierto de acción, de fuentes, cumplimiento de
`debe_contener` y de `no_debe_contener` (matching de `matching.py`, con alternativas «|»),
tasa de respuesta de respaldo (campo `respaldo` de la traza), costo total y latencia promedio.
Todo sale de las trazas de T17 (sink en memoria): tokens, costo y latencia. Las respuestas del
reporte van enmascaradas con `pii.py`.

Para los tests el runner recibe una fábrica `fabrica(caso, sink) -> Orquestador` inyectable (con
FakeLLM); nunca llama a la red por sí mismo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tiendahogar_agent.adaptadores.fabrica_llm import crear_llm
from tiendahogar_agent.config import Settings, cargar_settings
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.matching import faltantes, presentes
from tiendahogar_agent.orquestador import Orquestador
from tiendahogar_agent.pii import enmascarar_pii
from tiendahogar_agent.retriever import construir_retriever

RAIZ_REPO = Path(__file__).resolve().parents[2]
GOLDEN_POR_DEFECTO = RAIZ_REPO / "evals" / "golden_set.json"
RESULTADOS_POR_DEFECTO = RAIZ_REPO / "evals" / "resultados"
DOCS_POR_DEFECTO = RAIZ_REPO / "data" / "docs"
MAX_COSTO_POR_DEFECTO = 2.0
CODIGO_ABORTADO_POR_COSTO = 3

CATEGORIAS = (
    "politica", "pedido", "escalamiento", "fuera_de_alcance", "manipulacion", "hecho_critico",
)
# Fallos ya conocidos (id del caso -> motivo): cuentan como fallo normal, pero el reporte los marca.
# Hoy no hay ninguno: t18-17 dejó de serlo al acotar la tabla de hechos (ADR-009).
CASOS_CONOCIDOS: dict[str, str] = {}
ADVERTENCIA_PRECIOS = (
    "Precios por millón de tokens tomados de settings.yaml, NO verificados en vivo contra la "
    "página oficial: el costo del reporte es una estimación."
)
MENSAJE_OMITIDO = (
    "Evals con LLM real omitido: falta ANTHROPIC_API_KEY (defínela en el entorno o en .env). "
    "No se llamó a ninguna API."
)

FabricaOrquestador = Callable[[dict[str, Any], Any], Orquestador]


class SinkMemoria:
    """TraceSink en memoria (el runner lee de aquí la traza de cada turno)."""

    def __init__(self) -> None:
        self.trazas: list[dict[str, Any]] = []

    def registrar(self, traza: dict[str, Any]) -> None:
        self.trazas.append(traza)


# ------------------------------------------------------------------ evaluación de un caso
def cargar_golden(ruta: Path = GOLDEN_POR_DEFECTO) -> list[dict[str, Any]]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def evaluar_caso(caso: dict[str, Any], respuesta: Any, traza: dict[str, Any]) -> dict[str, Any]:
    """Compara la respuesta del agente con las expectativas del caso."""
    texto = respuesta.respuesta
    faltan = faltantes(texto, caso["debe_contener"])
    prohibidas = presentes(texto, caso["no_debe_contener"])
    obtenidas = list(respuesta.fuentes)
    resultado: dict[str, Any] = {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "accion_esperada": caso["accion_esperada"],
        "accion_obtenida": respuesta.accion,
        "accion_ok": respuesta.accion == caso["accion_esperada"],
        "fuentes_esperadas": list(caso["fuentes_esperadas"]),
        "fuentes_obtenidas": obtenidas,
        "fuentes_ok": set(obtenidas) == set(caso["fuentes_esperadas"]),
        "debe_ok": not faltan,
        "debe_faltantes": faltan,
        "no_debe_ok": not prohibidas,
        "no_debe_presentes": prohibidas,
        "canal": respuesta.canal,
        "respaldo": bool(traza.get("respaldo", False)),
        "reglas_fallidas": list(traza.get("reglas_fallidas", [])),
        "tokens_entrada": int(traza.get("tokens_entrada", 0)),
        "tokens_salida": int(traza.get("tokens_salida", 0)),
        "costo_usd": float(traza.get("costo_usd", 0.0)),
        "latencia_ms": float(traza.get("latencia_ms", 0.0)),
        "conocido": caso["id"] in CASOS_CONOCIDOS,
        "respuesta": enmascarar_pii(texto),
    }
    if resultado["conocido"]:
        resultado["motivo_conocido"] = CASOS_CONOCIDOS[caso["id"]]
    resultado["caso_ok"] = all(
        resultado[k] for k in ("accion_ok", "fuentes_ok", "debe_ok", "no_debe_ok")
    )
    return resultado


# ------------------------------------------------------------------ métricas
def _tasa(valores: list[bool]) -> float:
    return round(sum(valores) / len(valores), 4) if valores else 0.0


def metricas(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Métricas de un grupo de resultados (una categoría o todos)."""
    n = len(resultados)
    return {
        "n": n,
        "accion_acierto": _tasa([r["accion_ok"] for r in resultados]),
        "fuentes_acierto": _tasa([r["fuentes_ok"] for r in resultados]),
        "debe_contener_cumplimiento": _tasa([r["debe_ok"] for r in resultados]),
        "no_debe_contener_cumplimiento": _tasa([r["no_debe_ok"] for r in resultados]),
        "caso_ok": _tasa([r["caso_ok"] for r in resultados]),
        "respaldo_tasa": _tasa([r["respaldo"] for r in resultados]),
        "costo_total_usd": round(sum(r["costo_usd"] for r in resultados), 6),
        "latencia_promedio_ms": round(sum(r["latencia_ms"] for r in resultados) / n, 3) if n else 0.0,
    }


def metricas_por_categoria(resultados: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    orden = [c for c in CATEGORIAS if any(r["categoria"] == c for r in resultados)]
    orden += sorted({r["categoria"] for r in resultados} - set(CATEGORIAS))
    return {c: metricas([r for r in resultados if r["categoria"] == c]) for c in orden}


# ------------------------------------------------------------------ ejecución
def ejecutar_evals(
    casos: list[dict[str, Any]],
    fabrica: FabricaOrquestador,
    settings: Settings,
    max_costo: float = MAX_COSTO_POR_DEFECTO,
    ahora: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Corre los casos y arma el reporte. Si el costo acumulado supera `max_costo`, se detiene."""
    ahora = ahora or _dt.datetime.now().astimezone()
    sink = SinkMemoria()
    resultados: list[dict[str, Any]] = []
    abortado, motivo = False, ""
    for caso in casos:
        orquestador = fabrica(caso, sink)
        mensajes = caso["mensajes"]
        previas = len(sink.trazas)
        respuesta = orquestador.procesar(mensajes[-1]["content"], historial=mensajes[:-1] or None)
        traza = sink.trazas[-1] if len(sink.trazas) > previas else {}
        resultados.append(evaluar_caso(caso, respuesta, traza))
        acumulado = sum(r["costo_usd"] for r in resultados)
        if acumulado > max_costo:
            abortado = True
            motivo = (
                f"Se superó el tope de costo (--max-costo {max_costo:.2f} USD): "
                f"acumulado {acumulado:.4f} USD tras {len(resultados)} de {len(casos)} casos."
            )
            break
    return {
        "modelo": settings.llm_model,
        "fecha": ahora.isoformat(timespec="seconds"),
        "golden_casos": len(casos),
        "casos_ejecutados": len(resultados),
        "abortado": abortado,
        "motivo_aborto": motivo,
        "max_costo_usd": max_costo,
        "precios": {
            "entrada_por_millon_usd": settings.precio_entrada_por_millon,
            "salida_por_millon_usd": settings.precio_salida_por_millon,
        },
        "advertencia": ADVERTENCIA_PRECIOS,
        "global": metricas(resultados),
        "por_categoria": metricas_por_categoria(resultados),
        "casos": resultados,
    }


# ------------------------------------------------------------------ reporte
def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _fila_metricas(nombre: str, m: dict[str, Any]) -> str:
    return (
        f"| {nombre} | {m['n']} | {_pct(m['caso_ok'])} | {_pct(m['accion_acierto'])} | "
        f"{_pct(m['fuentes_acierto'])} | {_pct(m['debe_contener_cumplimiento'])} | "
        f"{_pct(m['no_debe_contener_cumplimiento'])} | {_pct(m['respaldo_tasa'])} | "
        f"{m['costo_total_usd']:.4f} | {m['latencia_promedio_ms']:.0f} |"
    )


def _celda(texto: str) -> str:
    return enmascarar_pii(texto).replace("|", "/").replace("\n", " ")


def render_markdown(reporte: dict[str, Any]) -> str:
    """Resumen legible con tablas por categoría (sin claves ni PII)."""
    p = reporte["precios"]
    L = [
        f"# Reporte de evals — {reporte['modelo']}",
        "",
        f"- Fecha: {reporte['fecha']}",
        f"- Modelo: `{reporte['modelo']}`",
        f"- Golden: {reporte['golden_casos']} casos (ejecutados: {reporte['casos_ejecutados']})",
        (
            f"- Precios usados (USD por millón de tokens): entrada {p['entrada_por_millon_usd']}, "
            f"salida {p['salida_por_millon_usd']}"
        ),
        f"- Tope de costo: {reporte['max_costo_usd']:.2f} USD",
        f"- **Advertencia:** {reporte['advertencia']}",
    ]
    if reporte["abortado"]:
        L += ["", f"> **ABORTADO.** {reporte['motivo_aborto']} El reporte es parcial."]
    L += [
        "",
        "## Métricas por categoría",
        "",
        (
            "| Categoría | Casos | Caso OK | Acción | Fuentes | debe_contener | no_debe_contener "
            "| Respaldo | Costo (USD) | Latencia prom. (ms) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cat, m in reporte["por_categoria"].items():
        L.append(_fila_metricas(cat, m))
    L.append(_fila_metricas("**Global**", reporte["global"]))
    L += [
        "",
        (
            "«Caso OK» exige acción, fuentes, debe_contener y no_debe_contener a la vez. "
            "«Respaldo» es la tasa de turnos resueltos con una plantilla de respaldo "
            "(verificación de salida fallida o fallo seguro)."
        ),
        "",
        "## Casos",
        "",
        (
            "| Id | Categoría | Acción (obt./esp.) | Fuentes | debe | no_debe | Respaldo "
            "| Costo (USD) | Latencia (ms) | Nota |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in reporte["casos"]:
        nota = "conocido" if c["conocido"] and not c["caso_ok"] else ""
        L.append(
            f"| {c['id']} | {c['categoria']} | {c['accion_obtenida']}/{c['accion_esperada']} | "
            f"{'ok' if c['fuentes_ok'] else 'NO'} | {'ok' if c['debe_ok'] else 'NO'} | "
            f"{'ok' if c['no_debe_ok'] else 'NO'} | {'sí' if c['respaldo'] else 'no'} | "
            f"{c['costo_usd']:.4f} | {c['latencia_ms']:.0f} | {nota} |"
        )
    fallidos = [c for c in reporte["casos"] if not c["caso_ok"]]
    L += ["", "## Casos que fallan", ""]
    if not fallidos:
        L.append("Ninguno.")
    for c in fallidos:
        marca = " (conocido)" if c["conocido"] else ""
        L += [
            f"### {c['id']}{marca}",
            "",
            f"- Acción: obtenida `{c['accion_obtenida']}`, esperada `{c['accion_esperada']}`",
            f"- Fuentes: obtenidas {c['fuentes_obtenidas']}, esperadas {c['fuentes_esperadas']}",
            f"- debe_contener que faltaron: {c['debe_faltantes'] or 'ninguno'}",
            f"- no_debe_contener presentes: {c['no_debe_presentes'] or 'ninguno'}",
            (
                f"- Respaldo: {'sí' if c['respaldo'] else 'no'}; reglas fallidas: "
                f"{c['reglas_fallidas'] or 'ninguna'}"
            ),
            f"- Respuesta (PII enmascarada): {_celda(c['respuesta'])}",
        ]
        if c["conocido"]:
            L.append(f"- Motivo conocido: {c['motivo_conocido']}")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def _slug(modelo: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", modelo).strip("-") or "modelo"


def escribir_reporte(
    reporte: dict[str, Any], carpeta: Path, ahora: _dt.datetime | None = None
) -> tuple[Path, Path]:
    """Guarda `AAAA-MM-DD-HHMM-<modelo>.json` y `.md` en `carpeta` (se crea si falta)."""
    ahora = ahora or _dt.datetime.now().astimezone()
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    base = f"{ahora:%Y-%m-%d-%H%M}-{_slug(reporte['modelo'])}"
    ruta_json, ruta_md = carpeta / f"{base}.json", carpeta / f"{base}.md"
    ruta_json.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    ruta_md.write_text(render_markdown(reporte), encoding="utf-8", newline="\n")
    return ruta_json, ruta_md


# ------------------------------------------------------------------ CLI
def hay_api_key(settings: Settings) -> bool:
    clave = settings.anthropic_api_key
    return clave is not None and bool(clave.get_secret_value().strip())


def _fabrica_real(settings: Settings, docs: Path = DOCS_POR_DEFECTO) -> FabricaOrquestador:
    """LLM real + retriever híbrido (embedder local), construidos una sola vez."""
    chunks = FileSystemDocumentSource.desde_settings(docs, settings).cargar()
    llm = crear_llm(settings)
    retriever = construir_retriever(settings, chunks)

    def fabrica(caso: dict[str, Any], sink: Any) -> Orquestador:
        return Orquestador(llm, retriever, settings, trace_sink=sink)

    return fabrica


def _parsear(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m tiendahogar_agent.evals",
        description="Ejecuta el golden set con el LLM real (solo con ANTHROPIC_API_KEY).",
    )
    p.add_argument("--golden", type=Path, default=GOLDEN_POR_DEFECTO)
    p.add_argument("--salida", type=Path, default=RESULTADOS_POR_DEFECTO,
                   help="carpeta del reporte (por defecto evals/resultados)")
    p.add_argument("--max-costo", type=float, default=MAX_COSTO_POR_DEFECTO,
                   help="tope de costo en USD; se aborta al superarlo (por defecto 2)")
    return p.parse_args(argv)


def principal(
    argv: Sequence[str] | None = None,
    settings: Settings | None = None,
    fabrica: FabricaOrquestador | None = None,
) -> int:
    """Punto de entrada. `settings` y `fabrica` son inyectables (tests); sin ellos: real."""
    args = _parsear(argv)
    settings = settings if settings is not None else cargar_settings()
    if fabrica is None:
        if not hay_api_key(settings):
            print(MENSAJE_OMITIDO)
            return 0
        settings = settings.model_copy(update={"llm_provider": "anthropic"})
        fabrica = _fabrica_real(settings)
    casos = cargar_golden(args.golden)
    print(f"Ejecutando {len(casos)} casos con el modelo {settings.llm_model} ...")
    reporte = ejecutar_evals(casos, fabrica, settings, max_costo=args.max_costo)
    ruta_json, ruta_md = escribir_reporte(reporte, args.salida)
    g = reporte["global"]
    print(
        f"Global: {reporte['casos_ejecutados']}/{reporte['golden_casos']} casos, "
        f"caso OK {_pct(g['caso_ok'])}, acción {_pct(g['accion_acierto'])}, "
        f"fuentes {_pct(g['fuentes_acierto'])}, respaldo {_pct(g['respaldo_tasa'])}, "
        f"costo {g['costo_total_usd']:.4f} USD, latencia prom. {g['latencia_promedio_ms']:.0f} ms"
    )
    for cat, m in reporte["por_categoria"].items():
        print(f"  {cat}: {m['n']} casos, caso OK {_pct(m['caso_ok'])}")
    print(f"Reporte: {ruta_json}\nResumen: {ruta_md}")
    if reporte["abortado"]:
        print(f"ABORTADO por tope de costo. {reporte['motivo_aborto']}")
        return CODIGO_ABORTADO_POR_COSTO
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(principal())
