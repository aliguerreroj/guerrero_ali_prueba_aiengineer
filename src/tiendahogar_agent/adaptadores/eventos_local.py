"""Adaptador local de `EventBus` (T22): memoria + JSONL opcional, ignora duplicados.

Un evento con `idempotency_key` ya vista (en esta instancia o en el archivo existente) se
descarta. Cada evento nuevo se agrega como una línea JSON. Thread-safe dentro del proceso
(`threading.Lock`); no coordina varios procesos (en producción lo haría el broker, ver ADR-010).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOMBRE_ARCHIVO = "eventos_escalamiento.jsonl"


class LocalEventBus:
    """Bus local idempotente. `ruta=None`: solo memoria."""

    def __init__(self, ruta: Path | None = None) -> None:
        self._ruta = Path(ruta) if ruta is not None else None
        self._lock = threading.Lock()
        self._claves: set[str] = set()
        self.eventos: list[dict[str, Any]] = []
        if self._ruta is not None and self._ruta.exists():
            self._cargar_claves(self._ruta)

    @property
    def ruta(self) -> Path | None:
        return self._ruta

    def _cargar_claves(self, ruta: Path) -> None:
        with ruta.open(encoding="utf-8") as f:
            for linea in f:
                try:
                    clave = json.loads(linea).get("idempotency_key")
                except (ValueError, AttributeError):
                    continue  # línea corrupta: se ignora
                if isinstance(clave, str):
                    self._claves.add(clave)

    def _termina_a_medias(self) -> bool:
        """True si el archivo existe y su última línea quedó sin salto (escritura truncada)."""
        try:
            with self._ruta.open("rb") as f:
                f.seek(0, 2)
                if f.tell() == 0:
                    return False
                f.seek(-1, 2)
                return f.read(1) != b"\n"
        except OSError:
            return False

    def publicar(self, evento: dict[str, Any]) -> None:
        clave = evento.get("idempotency_key")
        with self._lock:
            if isinstance(clave, str):
                if clave in self._claves:
                    logger.info("evento duplicado ignorado clave=%s", clave[:12])
                    return
                self._claves.add(clave)
            if self._ruta is not None:
                try:
                    self._ruta.parent.mkdir(parents=True, exist_ok=True)
                    prefijo = "\n" if self._termina_a_medias() else ""
                    with self._ruta.open("a", encoding="utf-8", newline="\n") as f:
                        f.write(prefijo + json.dumps(evento, ensure_ascii=False) + "\n")
                except OSError:
                    if isinstance(clave, str):
                        self._claves.discard(clave)  # no se dio por publicado
                    raise
            self.eventos.append(dict(evento))
