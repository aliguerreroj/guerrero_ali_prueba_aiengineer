"""Trazas por turno (T17): acumulador, costo y construcción de la línea JSONL.

El orquestador abre una `TrazaTurno` por turno en un `ContextVar` (sin estado compartido entre
turnos ni entre hilos). Las llamadas al LLM pasan por `LLMContado`, que suma los tokens de
TODAS las llamadas del turno sin que el dominio lo sepa. Nada de lo que se guarda contiene el
mensaje del cliente ni la respuesta; los argumentos de tools se enmascaran con `pii.py`.

Costo = entrada/1e6*precio_entrada + salida/1e6*precio_salida (USD, precios en settings.yaml).
Con `llm_provider=fake` el costo sale de los tokens que declare el doble (0 con LLMDemo o con un
FakeLLM sin tokens); `costo_es_estimacion` es true si el modelo configurado no es el de los precios.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from tiendahogar_agent.config import Settings
from tiendahogar_agent.models import AgentResponse, LLMResponse
from tiendahogar_agent.pii import enmascarar_pii
from tiendahogar_agent.retriever import ResultadoRecuperacion

MODELO_PRECIOS = "claude-haiku-4-5-20251001"


@dataclass
class TrazaTurno:
    """Datos recolectados durante un turno."""

    tokens_entrada: int = 0
    tokens_salida: int = 0
    llamadas_llm: int = 0
    categoria: str | None = None
    regla: str | None = None
    documentos: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    reglas_fallidas: list[str] = field(default_factory=list)
    reintentos: int = 0
    # True si el turno se resolvió con una plantilla/mensaje de respaldo (verificación de salida
    # fallida o fallo seguro), no con un escalamiento redactado por el LLM (T19).
    respaldo: bool = False

    def sumar_uso(self, respuesta: LLMResponse) -> None:
        self.llamadas_llm += 1
        self.tokens_entrada += respuesta.uso.entrada
        self.tokens_salida += respuesta.uso.salida

    def agregar_documentos(self, resultados: list[ResultadoRecuperacion], origen: str) -> None:
        for r in resultados:
            self.documentos.append({
                "doc_id": r.fuente.doc_id,
                "puntaje_bm25": r.puntaje_bm25,
                "similitud": r.similitud,
                "score": r.score,
                "origen": origen,
            })

    def agregar_tool(self, nombre: str, argumentos: Any, origen: str = "llm") -> None:
        self.tools.append({
            "nombre": enmascarar_pii(str(nombre)),
            "argumentos": _enmascarar(argumentos),
            "origen": origen,
        })


_ACTUAL: contextvars.ContextVar[TrazaTurno | None] = contextvars.ContextVar(
    "traza_turno", default=None
)


def traza_actual() -> TrazaTurno | None:
    return _ACTUAL.get()


def marcar_respaldo() -> None:
    """Marca el turno en curso como resuelto con una plantilla de respaldo (si hay traza)."""
    traza = _ACTUAL.get()
    if traza is not None:
        traza.respaldo = True


@contextlib.contextmanager
def traza_de_turno(activa: bool = True) -> Iterator[TrazaTurno | None]:
    """Abre la traza del turno en el contexto actual (propia de este hilo/turno)."""
    if not activa:
        yield None
        return
    traza = TrazaTurno()
    token = _ACTUAL.set(traza)
    try:
        yield traza
    finally:
        _ACTUAL.reset(token)


class LLMContado:
    """Envuelve un LLMClient y suma el uso de cada respuesta a la traza del turno en curso."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def completar(self, *args: Any, **kwargs: Any) -> LLMResponse:
        respuesta = self._llm.completar(*args, **kwargs)
        traza = _ACTUAL.get()
        if traza is not None:
            traza.sumar_uso(respuesta)
        return respuesta


def _enmascarar(valor: Any) -> Any:
    if isinstance(valor, str):
        return enmascarar_pii(valor)
    if isinstance(valor, dict):
        return {enmascarar_pii(str(k)): _enmascarar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_enmascarar(v) for v in valor]
    if valor is None or isinstance(valor, (bool, int, float)):
        return valor
    return enmascarar_pii(str(valor))


def calcular_costo(tokens_entrada: int, tokens_salida: int, settings: Settings) -> float:
    return (
        tokens_entrada / 1e6 * settings.precio_entrada_por_millon
        + tokens_salida / 1e6 * settings.precio_salida_por_millon
    )


def construir_traza(
    respuesta: AgentResponse, traza: TrazaTurno, settings: Settings, latencia_ms: float
) -> dict[str, Any]:
    """Línea de traza (sin mensaje ni respuesta del cliente)."""
    return {
        "trace_id": respuesta.trace_id,
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        "accion": respuesta.accion,
        "categoria": traza.categoria,
        "regla": traza.regla,
        "documentos": traza.documentos,
        "tools": traza.tools,
        "reglas_fallidas": traza.reglas_fallidas,
        "reintentos": traza.reintentos,
        "respaldo": traza.respaldo,
        "llamadas_llm": traza.llamadas_llm,
        "tokens_entrada": traza.tokens_entrada,
        "tokens_salida": traza.tokens_salida,
        "costo_usd": calcular_costo(traza.tokens_entrada, traza.tokens_salida, settings),
        "modelo": settings.llm_model,
        "costo_es_estimacion": settings.llm_model != MODELO_PRECIOS,
        "latencia_ms": round(latencia_ms, 3),
    }
