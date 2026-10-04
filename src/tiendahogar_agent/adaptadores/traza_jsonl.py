"""Adaptador de `TraceSink` (T17): una línea JSON por turno, en modo append, UTF-8."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

NOMBRE_ARCHIVO = "trazas.jsonl"


class JsonlTraceSink:
    """Escribe cada traza como una línea de `<carpeta>/trazas.jsonl` (carpeta creada al registrar).

    Thread-safe: un Lock serializa la escritura, así que las líneas nunca se mezclan.
    """

    def __init__(self, carpeta: Path) -> None:
        self._ruta = Path(carpeta) / NOMBRE_ARCHIVO
        self._lock = threading.Lock()

    @property
    def ruta(self) -> Path:
        return self._ruta

    def registrar(self, traza: dict[str, Any]) -> None:
        linea = json.dumps(traza, ensure_ascii=False) + "\n"
        with self._lock:
            self._ruta.parent.mkdir(parents=True, exist_ok=True)
            with self._ruta.open("a", encoding="utf-8", newline="\n") as f:
                f.write(linea)
