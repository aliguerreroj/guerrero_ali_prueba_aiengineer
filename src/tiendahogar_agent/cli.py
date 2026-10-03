"""CLI de chat (T16): conversa con el agente en la terminal.

Comando (con el entorno activado y el paquete instalado con `python -m pip install -e ".[dev]"`):

    python -m tiendahogar_agent.cli

Alternativa equivalente si el paquete está instalado: `tiendahogar-chat`.
Para una sola consulta o para scripts:

    python -m tiendahogar_agent.cli --una-vez "¿Cuánto dura la garantía?"
    echo "¿Dónde está mi pedido ORD-1001?" | python -m tiendahogar_agent.cli

Se sale con «salir», «exit», Ctrl+D o Ctrl+C. Con stdin no interactivo, cada línea es un turno.

Proveedor (variable `LLM_PROVIDER` o `.env`):
- `fake` (por defecto, sin red ni API key): `LLMDemo` (reglas fijas, ver `llm_demo.py`) y retriever
  solo léxico (BM25; no descarga ningún modelo). Las respuestas son sencillas pero reales: citan
  los documentos o la tabla de pedidos.
- `anthropic` / `azure`: LLM real con la clave del entorno y retriever híbrido (BM25 + embeddings
  de fastembed; la primera vez descarga el modelo, por lo que el arranque tarda más).

El historial de la sesión lo acumula la CLI (turnos `user`/`assistant`, máximo
`MAX_MENSAJES_HISTORIAL`). Tras cada respuesta se muestra la acción, las fuentes, el canal (si
escala) y el trace_id. Los logs (WARNING en adelante) van a
`logs/tiendahogar.log` (raíz del repo, o `TIENDAHOGAR_LOGS_DIR`); el cliente no los ve. Con `--debug`
también salen por stderr y el nivel baja a DEBUG.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.adaptadores.fabrica_llm import crear_llm
from tiendahogar_agent.config import Settings, cargar_settings
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.llm_demo import UMBRAL_SEMANTICO_IMPOSIBLE, EmbedderConstante, LLMDemo
from tiendahogar_agent.models import AgentResponse
from tiendahogar_agent.orquestador import MAX_MENSAJES_HISTORIAL, Orquestador
from tiendahogar_agent.retriever import Retriever, construir_retriever

RAIZ_REPO = Path(__file__).resolve().parents[2]
DOCS_POR_DEFECTO = RAIZ_REPO / "data" / "docs"
PALABRAS_SALIDA = {"salir", "exit"}
PROMPT = "Tú> "
BIENVENIDA = "TiendaHogar · asistente de soporte. Escribe «salir» para terminar."
DESPEDIDA = "¡Hasta pronto!"


def construir_orquestador(settings: Settings, docs: Path = DOCS_POR_DEFECTO) -> Orquestador:
    """Orquestador real. `fake`: LLMDemo + retriever léxico; otro proveedor: LLM y retriever híbrido."""
    chunks = FileSystemDocumentSource.desde_settings(docs, settings).cargar()
    if settings.llm_provider == "fake":
        llm = LLMDemo()
        retriever = Retriever(
            chunks, IndiceLexico(chunks), EmbedderConstante(), InMemoryVectorStore(),
            top_k=settings.top_k, umbral_bm25=settings.umbral_bm25,
            umbral_semantico=UMBRAL_SEMANTICO_IMPOSIBLE,
        )
    else:
        llm = crear_llm(settings)  # ValueError claro (sin secretos) si falta la clave
        retriever = construir_retriever(settings, chunks)
    return Orquestador(llm, retriever, settings)


def formatear_respuesta(r: AgentResponse) -> str:
    """Respuesta y, debajo, acción, fuentes, canal (si hay) y trace_id."""
    lineas = [
        f"Agente: {r.respuesta}",
        f"  [acción: {r.accion}]",
        f"  [fuentes: {', '.join(r.fuentes) if r.fuentes else 'ninguna'}]",
    ]
    if r.canal:
        lineas.append(f"  [canal: {r.canal}]")
    lineas.append(f"  [trace_id: {r.trace_id}]")
    return "\n".join(lineas)


def _error_configuracion(exc: Exception) -> str:
    """Mensaje sin traza ni valores (por si alguno fuera un secreto)."""
    if isinstance(exc, ValidationError):
        campos = ", ".join(sorted({".".join(map(str, e["loc"])) for e in exc.errors()}))
        return f"Configuración inválida en: {campos}. Revisa tu .env y settings.yaml."
    if isinstance(exc, ValueError):
        return f"Configuración incompleta o inválida: {exc}"
    return (
        f"No pude iniciar el agente ({type(exc).__name__}). "
        "Revisa la configuración y los documentos."
    )


def _parsear(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tiendahogar-chat", description="Chat de soporte TiendaHogar.")
    p.add_argument("--una-vez", metavar="MENSAJE", help="procesa un solo mensaje y termina")
    p.add_argument("--docs", type=Path, default=DOCS_POR_DEFECTO, help="carpeta de documentos")
    p.add_argument(
        "--debug", action="store_true",
        help="muestra los logs en la consola (stderr) y baja el nivel a DEBUG (también en el archivo)",
    )
    return p.parse_args(argv)


def directorio_logs() -> Path:
    """Carpeta de logs: `TIENDAHOGAR_LOGS_DIR`; si no, `logs/` en la raíz del repo (instalación
    editable, donde está pyproject.toml); si no hay repo, `logs/` bajo el directorio actual."""
    explicito = os.environ.get("TIENDAHOGAR_LOGS_DIR", "").strip()
    if explicito:
        return Path(explicito)
    if (RAIZ_REPO / "pyproject.toml").exists():
        return RAIZ_REPO / "logs"
    return Path.cwd() / "logs"


def configurar_logging(debug: bool, carpeta: Path, consola: TextIO) -> Callable[[], None]:
    """Instala los handlers del CLI en el logger raíz y devuelve la función que los retira.

    - Archivo `<carpeta>/tiendahogar.log` (WARNING; DEBUG con `debug`). Si la carpeta no se puede
      crear o escribir, se degrada sin caer: no hay log a archivo.
    - Consola (`consola`, stderr) solo con `debug`: el cliente nunca ve WARNING/ERROR ni trazas.
    Es idempotente: antes de instalar retira los handlers propios previos (no se acumulan).
    """
    raiz = logging.getLogger("")
    paquete = logging.getLogger("tiendahogar_agent")
    for h in [h for h in raiz.handlers if getattr(h, "_tiendahogar_cli", False)]:
        raiz.removeHandler(h)
        h.close()
    nivel = logging.DEBUG if debug else logging.WARNING
    formato = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = []
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(carpeta / "tiendahogar.log", encoding="utf-8"))
    except OSError:
        pass  # sin escritura: se degrada a no loguear a archivo
    if debug:
        handlers.append(logging.StreamHandler(consola))
    if not handlers:
        handlers.append(logging.NullHandler())  # evita el handler de último recurso (stderr)
    for h in handlers:
        h.setLevel(nivel)
        h.setFormatter(formato)
        h._tiendahogar_cli = True  # type: ignore[attr-defined]
        raiz.addHandler(h)
    nivel_previo_raiz, nivel_previo_paquete = raiz.level, paquete.level
    raiz.setLevel(logging.WARNING)
    paquete.setLevel(nivel)  # DEBUG solo para el paquete (no para librerías de terceros)

    def cerrar() -> None:
        for h in handlers:
            raiz.removeHandler(h)
            h.close()
        raiz.setLevel(nivel_previo_raiz)
        paquete.setLevel(nivel_previo_paquete)

    return cerrar


def main(
    argv: Sequence[str] | None = None,
    entrada: TextIO | None = None,
    salida: TextIO | None = None,
    orquestador: Any | None = None,
    errores: TextIO | None = None,
) -> int:
    """Punto de entrada testeable. Devuelve el código de salida (0 ok, 2 error de configuración)."""
    entrada = entrada if entrada is not None else sys.stdin
    salida = salida if salida is not None else sys.stdout
    errores = errores if errores is not None else sys.stderr
    args = _parsear(argv)
    cerrar_logging = configurar_logging(args.debug, directorio_logs(), errores)
    try:
        return _ejecutar(args, entrada, salida, errores, orquestador)
    finally:
        cerrar_logging()


def _ejecutar(
    args: argparse.Namespace, entrada: TextIO, salida: TextIO, errores: TextIO, orquestador: Any | None
) -> int:
    if orquestador is None:
        try:
            orquestador = construir_orquestador(cargar_settings(), args.docs)
        except Exception as exc:  # noqa: BLE001  borde de arranque: mensaje claro, sin traza
            print(_error_configuracion(exc), file=errores)
            return 2

    historial: list[dict[str, str]] = []

    def turno(mensaje: str) -> None:
        respuesta = orquestador.procesar(mensaje, list(historial))
        print(formatear_respuesta(respuesta), file=salida, flush=True)
        if mensaje.strip():  # los turnos vacíos no entran al historial
            historial.append({"role": "user", "content": mensaje})
            historial.append({"role": "assistant", "content": respuesta.respuesta})
            del historial[:-MAX_MENSAJES_HISTORIAL]
            while historial and historial[0]["role"] != "user":
                historial.pop(0)

    if args.una_vez is not None:
        try:
            turno(args.una_vez)
        except KeyboardInterrupt:
            print(file=salida)
        return 0
    interactivo = entrada.isatty()
    try:
        if interactivo:
            print(BIENVENIDA, file=salida, flush=True)
        while True:
            if interactivo:
                salida.write(PROMPT)
                salida.flush()
            linea = entrada.readline()
            if linea == "":  # EOF / Ctrl+D
                break
            mensaje = linea.rstrip("\r\n")
            if mensaje.strip().lower() in PALABRAS_SALIDA:
                break
            turno(mensaje)
    except KeyboardInterrupt:
        print(file=salida)
    if interactivo:
        print(DESPEDIDA, file=salida, flush=True)
    return 0


def _preparar_consola() -> None:
    """Salida segura en Windows: errors='replace' (y UTF-8 si no es terminal). El logging lo
    configura `main` (archivo; consola solo con --debug)."""
    for flujo in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            opciones: dict[str, str] = {"errors": "replace"}
            if not flujo.isatty():
                opciones["encoding"] = "utf-8"
            flujo.reconfigure(**opciones)


def principal() -> None:
    """Entry point de consola (`tiendahogar-chat`)."""
    _preparar_consola()
    sys.exit(main())


if __name__ == "__main__":
    principal()
