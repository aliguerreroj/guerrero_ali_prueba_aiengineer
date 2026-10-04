"""Evento de escalamiento (T22): modelo, clave de idempotencia determinista y construcción.

El evento NO lleva PII: ni el texto del cliente ni el `conversation_id` en claro. Solo
metadatos operativos (trace_id, categoría, canal) y la clave de idempotencia.

Clave de idempotencia = sha256 de campos estables, separados por «|» y con prefijo de versión:
    v1 | conversation_id | n.º de turno del cliente | huella(mensaje) | categoría | canal
- `huella(mensaje)`: sha256 del mensaje normalizado (NFKC, minúsculas, espacios colapsados);
  no se publica, solo alimenta el hash.
- Un reintento del MISMO turno (misma conversación, mismo historial y mismo mensaje) produce la
  MISMA clave aunque cambie el `trace_id` (un uuid nuevo por intento) y la hora.
- Casos distintos producen claves distintas: otra conversación, otro turno (n.º de mensajes del
  cliente en el historial), otro mensaje, otra categoría u otro canal.
- Sin `conversation_id` (CLI, evals) no hay identidad estable entre intentos: la base pasa a ser
  el `trace_id`, así que cada turno es un caso distinto (nunca se pierde un escalamiento).
Límite: si el turno ya se completó y el cliente repite el mismo mensaje DESPUÉS de que el
historial creció, el n.º de turno cambia y se considera un caso nuevo (lado seguro: no se
suprime un escalamiento legítimo).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

VERSION_ESQUEMA = 1
VERSION_CLAVE = "v1"
TIPO_EVENTO = "EscalationCreated"


class EscalationCreated(BaseModel):
    """Evento publicado cuando un turno termina con acción `escalar` (sin PII)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tipo: Literal["EscalationCreated"] = TIPO_EVENTO
    schema_version: int = VERSION_ESQUEMA
    trace_id: str
    categoria: str
    canal: str | None
    idempotency_key: str
    ocurrido_en: str


def huella_mensaje(mensaje: str) -> str:
    """sha256 del mensaje normalizado (solo para alimentar la clave; nunca se publica)."""
    normal = unicodedata.normalize("NFKC", mensaje).casefold()
    normal = re.sub(r"\s+", " ", normal).strip()
    return hashlib.sha256(normal.encode("utf-8")).hexdigest()


def calcular_clave_idempotencia(
    *,
    trace_id: str,
    categoria: str,
    canal: str | None,
    conversation_id: str | None = None,
    turno: int = 0,
    mensaje: str = "",
) -> str:
    """Clave determinista (sha256 hex) del caso de escalamiento. Ver el docstring del módulo."""
    if conversation_id:
        base = f"conv:{conversation_id}|{turno}|{huella_mensaje(mensaje)}"
    else:
        base = f"trace:{trace_id}"
    campos = [VERSION_CLAVE, base, categoria, canal or ""]
    return hashlib.sha256("|".join(campos).encode("utf-8")).hexdigest()


def construir_evento(
    *,
    trace_id: str,
    categoria: str,
    canal: str | None,
    conversation_id: str | None = None,
    turno: int = 0,
    mensaje: str = "",
) -> EscalationCreated:
    clave = calcular_clave_idempotencia(
        trace_id=trace_id, categoria=categoria, canal=canal,
        conversation_id=conversation_id, turno=turno, mensaje=mensaje,
    )
    ahora = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    return EscalationCreated(
        trace_id=trace_id, categoria=categoria, canal=canal, idempotency_key=clave,
        ocurrido_en=ahora,
    )
