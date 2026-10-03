"""Carga de documentos y chunking.

Las funciones por estrategia son puras (sin I/O). `FileSystemDocumentSource`
implementa el puerto `DocumentSource` leyendo `.md` en modo solo lectura.

Metadatos de cada chunk: `titulo`, `fuente` (nombre de archivo), `estrategia`,
`posicion` (índice base 0 dentro del documento) y `total_chunks`.
"""

from __future__ import annotations

import re
from pathlib import Path

from tiendahogar_agent.config import Settings
from tiendahogar_agent.models import Chunk

ESTRATEGIAS = ("none", "fixed", "recursive", "auto")
_SEPARADORES = ("\n\n", "\n", ". ", " ")


class ErrorDocumentos(Exception):
    """Error al cargar o trocear documentos."""


def _validar_estrategia(estrategia: str) -> None:
    if estrategia not in ESTRATEGIAS:
        raise ValueError(
            f"Estrategia de chunking inválida: {estrategia!r}. Opciones: {', '.join(ESTRATEGIAS)}."
        )


def _validar_params(tamano: int, solape: int) -> None:
    if tamano < 1:
        raise ValueError("El tamaño de chunk debe ser >= 1.")
    if not 0 <= solape < tamano:
        raise ValueError("El solape debe estar entre 0 y tamaño - 1.")


def _limpiar(trozos: list[str]) -> list[str]:
    return [t.strip() for t in trozos if t.strip()]


def chunk_none(texto: str) -> list[str]:
    """Un único chunk con todo el texto (lista vacía si no hay contenido)."""
    return _limpiar([texto])


def chunk_fixed(texto: str, tamano: int, solape: int) -> list[str]:
    """Ventanas de `tamano` caracteres desplazadas `tamano - solape`."""
    _validar_params(tamano, solape)
    paso = tamano - solape
    trozos = []
    i = 0
    while i < len(texto):
        trozos.append(texto[i : i + tamano])
        if i + tamano >= len(texto):
            break
        i += paso
    return _limpiar(trozos)


def _atomos(texto: str, tamano: int, separadores: tuple[str, ...]) -> list[str]:
    """Parte el texto en piezas <= tamano usando separadores de mayor a menor."""
    if len(texto) <= tamano:
        return [texto]
    if not separadores:
        return [texto[i : i + tamano] for i in range(0, len(texto), tamano)]
    sep, resto = separadores[0], separadores[1:]
    partes = texto.split(sep)
    piezas = [p + sep for p in partes[:-1]] + [partes[-1]]
    salida: list[str] = []
    for p in piezas:
        salida.extend(_atomos(p, tamano, resto) if len(p) > tamano else [p])
    return salida


def chunk_recursive(texto: str, tamano: int, solape: int) -> list[str]:
    """Divide respetando párrafos/líneas/frases/palabras; solape = cola del chunk previo."""
    _validar_params(tamano, solape)
    atomos = [a for a in _atomos(texto, tamano, _SEPARADORES) if a]
    chunks: list[list[str]] = []
    actual: list[str] = []
    longitud = 0
    for a in atomos:
        if actual and longitud + len(a) > tamano:
            chunks.append(actual)
            cola: list[str] = []
            suma = 0
            for previo in reversed(actual):
                if suma + len(previo) > solape:
                    break
                cola.insert(0, previo)
                suma += len(previo)
            if suma + len(a) > tamano:
                cola, suma = [], 0
            actual, longitud = cola, suma
        actual.append(a)
        longitud += len(a)
    if actual:
        chunks.append(actual)
    return _limpiar(["".join(c) for c in chunks])


def trocear(
    texto: str,
    estrategia: str = "auto",
    tamano: int = 500,
    solape: int = 50,
    umbral_corto: int = 1000,
) -> list[str]:
    """Aplica la estrategia. `auto`: none si len(texto) <= umbral_corto, si no recursive."""
    _validar_estrategia(estrategia)
    if estrategia == "auto":
        estrategia = "none" if len(texto.strip()) <= umbral_corto else "recursive"
    if estrategia == "none":
        return chunk_none(texto)
    if estrategia == "fixed":
        return chunk_fixed(texto, tamano, solape)
    return chunk_recursive(texto, tamano, solape)


def normalizar_saltos(texto: str) -> str:
    """CRLF/CR -> LF (solo en memoria)."""
    return texto.replace("\r\n", "\n").replace("\r", "\n")


def extraer_titulo(texto: str, nombre_archivo: str) -> str:
    """Texto del primer H1 (sin '# '). Sin H1: se deriva del nombre de archivo.

    Se elige derivar (no fallar) porque un documento sin encabezado sigue siendo
    útil; el título derivado es solo el nombre sin prefijo `docN_`, sin inventar contenido.
    """
    m = re.search(r"^#[ \t]+(.+?)[ \t]*$", texto, flags=re.MULTILINE)
    if m:
        return m.group(1)
    base = re.sub(r"^doc\d+_", "", Path(nombre_archivo).stem)
    return base.replace("_", " ").strip() or Path(nombre_archivo).stem


def derivar_doc_id(nombre_archivo: str) -> str:
    """`doc1_garantia.md` -> `doc1`; sin prefijo docN_, usa el nombre sin extensión."""
    stem = Path(nombre_archivo).stem
    m = re.match(r"^(doc\d+)_", stem)
    return m.group(1) if m else stem


def _clave_orden(ruta: Path) -> tuple[int, str]:
    m = re.match(r"^doc(\d+)_", ruta.name)
    return (int(m.group(1)) if m else 10**9, ruta.name)


class FileSystemDocumentSource:
    """Implementa `DocumentSource` leyendo los `.md` de un directorio (solo lectura)."""

    def __init__(
        self,
        directorio: str | Path,
        estrategia: str = "auto",
        tamano: int = 500,
        solape: int = 50,
        umbral_corto: int = 1000,
    ) -> None:
        _validar_estrategia(estrategia)
        _validar_params(tamano, solape)
        self._dir = Path(directorio)
        self._estrategia = estrategia
        self._tamano = tamano
        self._solape = solape
        self._umbral = umbral_corto

    @classmethod
    def desde_settings(
        cls, directorio: str | Path, settings: Settings
    ) -> FileSystemDocumentSource:
        return cls(
            directorio,
            estrategia=settings.chunk_strategy,
            tamano=settings.chunk_tamano,
            solape=settings.chunk_solape,
            umbral_corto=settings.chunk_umbral_corto,
        )

    def cargar(self) -> list[Chunk]:
        if not self._dir.is_dir():
            raise ErrorDocumentos(f"El directorio de documentos no existe: {self._dir}")
        rutas = sorted(self._dir.glob("*.md"), key=_clave_orden)
        if not rutas:
            raise ErrorDocumentos(f"No se encontraron documentos .md en: {self._dir}")
        chunks: list[Chunk] = []
        for ruta in rutas:
            texto = normalizar_saltos(ruta.read_text(encoding="utf-8")).strip()
            if not texto:
                raise ErrorDocumentos(f"El documento está vacío: {ruta.name}")
            titulo = extraer_titulo(texto, ruta.name)
            doc_id = derivar_doc_id(ruta.name)
            trozos = trocear(texto, self._estrategia, self._tamano, self._solape, self._umbral)
            for i, t in enumerate(trozos):
                chunks.append(
                    Chunk(
                        texto=t,
                        doc_id=doc_id,
                        metadatos={
                            "titulo": titulo,
                            "fuente": ruta.name,
                            "estrategia": self._estrategia,
                            "posicion": i,
                            "total_chunks": len(trozos),
                        },
                    )
                )
        return chunks
