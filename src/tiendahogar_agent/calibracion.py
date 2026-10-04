"""Calibración de los umbrales de relevancia del retriever (T19, ADR-004). NO llama al LLM.

Uso (con el entorno activado; usa el embedder real configurado, local, sin costo de API):

    python -m tiendahogar_agent.calibracion [--salida CARPETA]

Para cada pregunta mide el puntaje máximo de BM25 y de coseno contra los chunks de
`data/docs/` y cuántos chunks pasarían la regla de ADR-004 (`BM25 > umbral_bm25` O
`coseno > umbral_semantico`) con los umbrales vigentes. Grupos:

- `en_dominio`: preguntas del golden de categoría politica o hecho_critico (se responden con
  los documentos, así que necesitan al menos un chunk relevante).
- `fuera_de_dominio`: las del golden `fuera_de_alcance` más `evals/preguntas_fuera_de_dominio.json`
  (sin `limite`). No deberían traer chunks.
- `limite`: preguntas fuera de dominio cercanas al dominio (p. ej. «recomiéndame una lavadora»);
  se reportan aparte y no cuentan para el margen.
- `otras`: pedido, escalamiento y manipulación (no dependen del retriever para decidir); solo
  se reportan.

El margen de una señal es `min(en_dominio) - max(fuera_de_dominio)` de los puntajes máximos
(positivo = las distribuciones se separan). Para un umbral: `margen_inferior = min(en_dominio)
- umbral` (cuánto sobra para no perder aciertos) y `margen_superior = umbral -
max(fuera_de_dominio)` (cuánto sobra para no dejar pasar ruido).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.config import cargar_settings
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import Chunk
from tiendahogar_agent.puertos import Embedder
from tiendahogar_agent.retriever import Retriever, chunk_relevante

RAIZ_REPO = Path(__file__).resolve().parents[2]
GOLDEN_POR_DEFECTO = RAIZ_REPO / "evals" / "golden_set.json"
FUERA_POR_DEFECTO = RAIZ_REPO / "evals" / "preguntas_fuera_de_dominio.json"
RESULTADOS_POR_DEFECTO = RAIZ_REPO / "evals" / "resultados"
DOCS_POR_DEFECTO = RAIZ_REPO / "data" / "docs"

CATEGORIAS_EN_DOMINIO = frozenset({"politica", "hecho_critico"})
BARRIDO_SEMANTICO = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
BARRIDO_BM25 = (0.5, 1.0, 1.5, 2.0)


def _hoy() -> _dt.date:
    return _dt.datetime.now().astimezone().date()


def preguntas_del_golden(casos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Último mensaje del usuario de cada caso, con su categoría."""
    return [
        {"id": c["id"], "categoria": c["categoria"], "pregunta": c["mensajes"][-1]["content"]}
        for c in casos
    ]


def cargar_preguntas_fuera(ruta: Path = FUERA_POR_DEFECTO) -> list[dict[str, Any]]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _grupo_golden(categoria: str) -> str:
    if categoria in CATEGORIAS_EN_DOMINIO:
        return "en_dominio"
    return "fuera_de_dominio" if categoria == "fuera_de_alcance" else "otras"


def _estadisticos(valores: list[float]) -> dict[str, Any]:
    if not valores:
        return {"n": 0, "min": None, "mediana": None, "max": None}
    return {
        "n": len(valores), "min": min(valores),
        "mediana": statistics.median(valores), "max": max(valores),
    }


def _margen(en_dominio: dict[str, Any], fuera: dict[str, Any]) -> float | None:
    if en_dominio["min"] is None or fuera["max"] is None:
        return None
    return en_dominio["min"] - fuera["max"]


def _resumen_senal(
    preguntas: list[dict[str, Any]], campo: str, umbral: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    en = _estadisticos([p[campo] for p in preguntas
                        if p["grupo"] == "en_dominio" and p[campo] is not None])
    fuera = _estadisticos([p[campo] for p in preguntas
                           if p["grupo"] == "fuera_de_dominio" and p[campo] is not None])
    senal = {"en_dominio": en, "fuera_de_dominio": fuera, "margen": _margen(en, fuera)}
    umbrales = {
        "valor": umbral,
        "margen_inferior": None if en["min"] is None else en["min"] - umbral,
        "margen_superior": None if fuera["max"] is None else umbral - fuera["max"],
    }
    return senal, umbrales


def _pasa(p: dict[str, Any], umbral_bm25: float, umbral_semantico: float) -> bool:
    return chunk_relevante(p["max_bm25"], p["max_coseno"], umbral_bm25, umbral_semantico)


def calibrar(
    golden: list[dict[str, Any]],
    fuera: list[dict[str, Any]],
    chunks: list[Chunk],
    embedder: Embedder,
    umbral_bm25: float,
    umbral_semantico: float,
) -> dict[str, Any]:
    """Mide puntajes y relevancia de cada pregunta. `golden`: items con id, categoria, pregunta."""
    retriever = Retriever(
        chunks, IndiceLexico(chunks), embedder, InMemoryVectorStore(), top_k=max(len(chunks), 1),
        umbral_bm25=umbral_bm25, umbral_semantico=umbral_semantico,
    )
    entradas = [(p, _grupo_golden(p["categoria"]), p["categoria"], False) for p in golden]
    entradas += [
        (p, "limite" if p.get("limite") else "fuera_de_dominio", None, bool(p.get("limite")))
        for p in fuera
    ]
    preguntas: list[dict[str, Any]] = []
    for p, grupo, categoria, limite in entradas:
        puntajes = retriever.puntajes(p["pregunta"])
        sims = [s for _, _, s in puntajes if s is not None]
        mejor = max(puntajes, key=lambda t: (t[2] if t[2] is not None else -2.0), default=None)
        preguntas.append({
            "id": p["id"],
            "grupo": grupo,
            "categoria": categoria,
            "limite": limite,
            "pregunta": p["pregunta"],
            "max_bm25": max((b for _, b, _ in puntajes), default=0.0),
            "max_coseno": max(sims) if sims else None,
            "mejor_doc_coseno": mejor[0].doc_id if mejor and mejor[2] is not None else None,
            "chunks_relevantes": sum(
                chunk_relevante(b, s, umbral_bm25, umbral_semantico) for _, b, s in puntajes
            ),
        })
    equivocados = []
    for p in preguntas:
        if p["grupo"] == "en_dominio" and p["chunks_relevantes"] == 0:
            motivo = "en dominio sin ningún chunk relevante (se perdería el sustento)"
        elif p["grupo"] == "fuera_de_dominio" and p["chunks_relevantes"] > 0:
            motivo = (f"fuera de dominio pero pasa la regla con {p['chunks_relevantes']} "
                      "chunk(s) (ruido)")
        else:
            continue
        equivocados.append({
            "id": p["id"], "grupo": p["grupo"], "pregunta": p["pregunta"], "motivo": motivo,
            "max_bm25": p["max_bm25"], "max_coseno": p["max_coseno"],
        })
    coseno, umbral_sem = _resumen_senal(preguntas, "max_coseno", umbral_semantico)
    bm25, umbral_lex = _resumen_senal(preguntas, "max_bm25", umbral_bm25)
    estrictas = [p for p in preguntas if p["grupo"] in ("en_dominio", "fuera_de_dominio")]
    n_en = sum(p["grupo"] == "en_dominio" for p in preguntas)
    n_fuera = sum(p["grupo"] == "fuera_de_dominio" for p in preguntas)

    def _barrido(candidatos: tuple[float, ...], variable: str) -> list[dict[str, Any]]:
        filas = []
        for u in candidatos:
            ub, us = (u, umbral_semantico) if variable == "bm25" else (umbral_bm25, u)
            filas.append({
                "umbral": u,
                "dominio_sin_chunks": sum(
                    p["grupo"] == "en_dominio" and not _pasa(p, ub, us) for p in estrictas),
                "fuera_con_chunks": sum(
                    p["grupo"] == "fuera_de_dominio" and _pasa(p, ub, us) for p in estrictas),
            })
        return filas

    return {
        "fecha": _hoy().isoformat(),
        "umbrales_vigentes": {"umbral_bm25": umbral_bm25, "umbral_semantico": umbral_semantico},
        "chunks": len(chunks),
        "preguntas": preguntas,
        "lado_equivocado": equivocados,
        "limite": [
            {"id": p["id"], "pregunta": p["pregunta"], "max_bm25": p["max_bm25"],
             "max_coseno": p["max_coseno"], "chunks_relevantes": p["chunks_relevantes"]}
            for p in preguntas if p["grupo"] == "limite"
        ],
        "resumen": {
            "n": {
                "en_dominio": n_en, "fuera_de_dominio": n_fuera,
                "limite": sum(p["grupo"] == "limite" for p in preguntas),
                "otras": sum(p["grupo"] == "otras" for p in preguntas),
            },
            "coseno": coseno,
            "bm25": bm25,
            "umbral_semantico": umbral_sem,
            "umbral_bm25": umbral_lex,
            "barrido_umbral_semantico": _barrido(BARRIDO_SEMANTICO, "semantico"),
            "barrido_umbral_bm25": _barrido(BARRIDO_BM25, "bm25"),
        },
    }


# ------------------------------------------------------------------ reporte
def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_markdown(cal: dict[str, Any]) -> str:
    r, u = cal["resumen"], cal["umbrales_vigentes"]
    L = [
        f"# Calibración del umbral de relevancia — {cal['fecha']}",
        "",
        f"- Modelo de embeddings: `{cal.get('embedding_model', 'n/d')}`; chunks indexados: {cal['chunks']}",
        f"- Umbrales vigentes: BM25 > {u['umbral_bm25']}, coseno > {u['umbral_semantico']}",
        (
            f"- Preguntas: {r['n']['en_dominio']} en dominio (politica/hecho_critico), "
            f"{r['n']['fuera_de_dominio']} fuera de dominio, {r['n']['limite']} límite, "
            f"{r['n']['otras']} otras (pedido/escalamiento/manipulación; solo informativas)"
        ),
        (
            "- No se llamó al LLM. El margen es `min(en dominio) - max(fuera de dominio)` del "
            "puntaje máximo por pregunta."
        ),
        "",
        "## Distribución del puntaje máximo por pregunta",
        "",
        "| Señal | Grupo | n | mín | mediana | máx |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, clave in (("coseno", "coseno"), ("BM25", "bm25")):
        for g in ("en_dominio", "fuera_de_dominio"):
            e = r[clave][g]
            L.append(f"| {nombre} | {g} | {e['n']} | {_f(e['min'])} | {_f(e['mediana'])} | {_f(e['max'])} |")
    L += [
        "",
        "## Márgenes",
        "",
        "| Señal | Margen entre grupos | Umbral | Margen inferior (aciertos) | Margen superior (ruido) |",
        "|---|---|---|---|---|",
        (
            f"| coseno | {_f(r['coseno']['margen'])} | {r['umbral_semantico']['valor']} | "
            f"{_f(r['umbral_semantico']['margen_inferior'])} | "
            f"{_f(r['umbral_semantico']['margen_superior'])} |"
        ),
        (
            f"| BM25 | {_f(r['bm25']['margen'])} | {r['umbral_bm25']['valor']} | "
            f"{_f(r['umbral_bm25']['margen_inferior'])} | "
            f"{_f(r['umbral_bm25']['margen_superior'])} |"
        ),
        "",
        (
            "Un margen negativo significa que las distribuciones se solapan (ningún umbral "
            "separa todo con esa señal sola)."
        ),
        "",
        "## Barrido del umbral semántico (BM25 vigente)",
        "",
        "| Umbral coseno | En dominio sin chunks | Fuera de dominio con chunks |",
        "|---|---|---|",
    ]
    L += [f"| {b['umbral']} | {b['dominio_sin_chunks']} | {b['fuera_con_chunks']} |"
          for b in r["barrido_umbral_semantico"]]
    L += ["", "## Barrido del umbral BM25 (coseno vigente)", "",
          "| Umbral BM25 | En dominio sin chunks | Fuera de dominio con chunks |", "|---|---|---|"]
    L += [f"| {b['umbral']} | {b['dominio_sin_chunks']} | {b['fuera_con_chunks']} |"
          for b in r["barrido_umbral_bm25"]]
    L += ["", "## Casos del lado equivocado (umbrales vigentes)", ""]
    if not cal["lado_equivocado"]:
        L.append("Ninguno.")
    else:
        L += ["| Id | Motivo | BM25 máx | Coseno máx | Pregunta |", "|---|---|---|---|---|"]
        L += [f"| {c['id']} | {c['motivo']} | {_f(c['max_bm25'])} | {_f(c['max_coseno'])} | "
              f"{c['pregunta'].replace('|', '/')} |" for c in cal["lado_equivocado"]]
    L += ["", "## Preguntas límite (no cuentan para el margen)", ""]
    if cal["limite"]:
        L += ["| Id | BM25 máx | Coseno máx | Chunks relevantes | Pregunta |", "|---|---|---|---|---|"]
        L += [f"| {c['id']} | {_f(c['max_bm25'])} | {_f(c['max_coseno'])} | "
              f"{c['chunks_relevantes']} | {c['pregunta'].replace('|', '/')} |" for c in cal["limite"]]
    else:
        L.append("Ninguna.")
    L += ["", "## Detalle por pregunta", "",
          "| Id | Grupo | BM25 máx | Coseno máx | Mejor doc | Chunks relevantes |",
          "|---|---|---|---|---|---|"]
    L += [f"| {p['id']} | {p['grupo']} | {_f(p['max_bm25'])} | {_f(p['max_coseno'])} | "
          f"{p['mejor_doc_coseno'] or '—'} | {p['chunks_relevantes']} |" for p in cal["preguntas"]]
    return "\n".join(L) + "\n"


def escribir_calibracion(
    cal: dict[str, Any], carpeta: Path, hoy: _dt.date | None = None
) -> tuple[Path, Path]:
    """Guarda `calibracion-AAAA-MM-DD.json` y `.md` en `carpeta`."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    base = f"calibracion-{(hoy or _hoy()).isoformat()}"
    rj, rm = carpeta / f"{base}.json", carpeta / f"{base}.md"
    rj.write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    rm.write_text(render_markdown(cal), encoding="utf-8", newline="\n")
    return rj, rm


def principal(argv: Sequence[str] | None = None, embedder: Embedder | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tiendahogar_agent.calibracion",
        description="Mide puntajes BM25/coseno en y fuera de dominio (sin llamar al LLM).",
    )
    p.add_argument("--golden", type=Path, default=GOLDEN_POR_DEFECTO)
    p.add_argument("--fuera", type=Path, default=FUERA_POR_DEFECTO)
    p.add_argument("--salida", type=Path, default=RESULTADOS_POR_DEFECTO)
    args = p.parse_args(argv)
    settings = cargar_settings()
    chunks = FileSystemDocumentSource.desde_settings(DOCS_POR_DEFECTO, settings).cargar()
    if embedder is None:
        from tiendahogar_agent.adaptadores.fastembed_embedder import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(settings.embedding_model)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    cal = calibrar(
        preguntas_del_golden(golden), cargar_preguntas_fuera(args.fuera), chunks, embedder,
        settings.umbral_bm25, settings.umbral_semantico,
    )
    cal["embedding_model"] = settings.embedding_model
    if any(q["max_coseno"] is None for q in cal["preguntas"]):
        print("AVISO: el embedder falló; hay preguntas sin coseno (solo BM25).")
    rj, rm = escribir_calibracion(cal, args.salida)
    r = cal["resumen"]
    print(f"Coseno: margen {_f(r['coseno']['margen'])}, bm25: margen {_f(r['bm25']['margen'])}")
    print(f"Del lado equivocado: {len(cal['lado_equivocado'])}")
    print(f"Reporte: {rj}\nResumen: {rm}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(principal())
