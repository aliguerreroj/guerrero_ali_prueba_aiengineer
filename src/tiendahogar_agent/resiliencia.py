"""Resiliencia (T11): fallo seguro = escalar a un humano.

API pequeña que usará el orquestador (T14):
- `llamar_llm_seguro`: llama al LLM con el timeout de Settings; devuelve la
  LLMResponse o una RespuestaFalloSeguro (error del LLM o respuesta vacía).
- `recuperar_seguro`: ejecuta el retriever; fallo -> RespuestaFalloSeguro.
- `revisar_resultado_tool`: `error_interno` de la tool de pedidos -> RespuestaFalloSeguro.
- `respuesta_fallo_seguro`: constructor único de la respuesta de fallo.

Reintentos: NO hay bucle propio. El SDK reintenta con su backoff (`max_retries`, que
T12 toma de `Settings.max_reintentos_llm`) y los adaptadores traducen sus errores a
`excepciones.ErrorLLM`. Solo se capturan excepciones específicas; un error inesperado
(un bug) se propaga. Los mensajes de error se registran con la PII enmascarada.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from tiendahogar_agent.config import Settings
from tiendahogar_agent.excepciones import ErrorLLM, ErrorRecuperacion
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.models import AgentResponse, LLMResponse
from tiendahogar_agent.pii import enmascarar_pii
from tiendahogar_agent.retriever import ResultadoRecuperacion
from tiendahogar_agent.texto import es_vacio_visible

logger = logging.getLogger(__name__)

MotivoFallo = Literal[
    "llm_error",
    "llm_respuesta_vacia",
    "tool_error_interno",
    "retriever_error",
    "bucle_sin_respuesta",  # T14: iteraciones agotadas, responder duplicado o texto sin responder
    "error_inesperado",  # T14: excepción no prevista capturada en el borde del orquestador
]

# Sin cifras, sin fuentes y sin compromisos: no dispara la verificación de salida.
MENSAJE_FALLO_SEGURO = (
    "Lo siento, tuve un problema técnico al atender tu consulta y prefiero no darte un "
    "dato incorrecto. Para que te ayuden bien, escríbele a nuestro equipo humano a "
    f"{CANAL_ESCALAMIENTO} y con gusto te atenderán."
)

# Fallos del retriever que se tratan como recuperables (índice o datos inconsistentes).
_ERRORES_RETRIEVER = (ErrorRecuperacion, ValueError, KeyError, IndexError)


class RespuestaFalloSeguro(BaseModel):
    """Respuesta de escalamiento por fallo, con el motivo para trazas/métricas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motivo: MotivoFallo
    respuesta: AgentResponse


class _Recuperador(Protocol):
    def recuperar(self, consulta: str | None) -> list[ResultadoRecuperacion]: ...


def respuesta_fallo_seguro(motivo: MotivoFallo, trace_id: str = "") -> RespuestaFalloSeguro:
    """Único constructor de la respuesta de fallo seguro (escalar al canal humano)."""
    return RespuestaFalloSeguro(
        motivo=motivo,
        respuesta=AgentResponse(
            respuesta=MENSAJE_FALLO_SEGURO,
            accion="escalar",
            fuentes=[],
            canal=CANAL_ESCALAMIENTO,
            trace_id=trace_id,
        ),
    )


def fallar(
    motivo: MotivoFallo,
    trace_id: str,
    exc: BaseException | None = None,
    detalle: str = "",
) -> RespuestaFalloSeguro:
    """Registra el fallo con traceback, todo enmascarado, y devuelve el fallo seguro."""
    tipo = type(exc).__name__ if exc is not None else "-"
    mensaje = enmascarar_pii(str(exc) if exc is not None else detalle)
    if exc is not None and exc.__traceback__ is not None:
        # El traceback puede contener PII en el texto de la excepción: se enmascara entero.
        mensaje = enmascarar_pii(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )
    logger.error(
        "fallo seguro aplicado=escalar motivo=%s tipo=%s trace_id=%s detalle=%s",
        motivo, tipo, trace_id, mensaje,
    )
    return respuesta_fallo_seguro(motivo, trace_id)


def llamar_llm_seguro(
    llm: Any,
    mensajes: list[dict[str, Any]],
    settings: Settings,
    tools: list[dict[str, Any]] | None = None,
    trace_id: str = "",
    max_tokens: int | None = None,
) -> LLMResponse | RespuestaFalloSeguro:
    """Llama al LLM con `settings.timeout_llm_s`; error o respuesta vacía -> fallo seguro."""
    try:
        # max_tokens solo se envía si se pide: los dobles/clientes sin ese parámetro siguen valiendo.
        extra = {} if max_tokens is None else {"max_tokens": max_tokens}
        respuesta = llm.completar(mensajes, tools=tools, timeout=settings.timeout_llm_s, **extra)
    except ErrorLLM as exc:
        return fallar("llm_error", trace_id, exc)
    if es_vacio_visible(respuesta.texto) and not respuesta.llamadas_tools:
        return fallar("llm_respuesta_vacia", trace_id, detalle="el LLM no devolvió texto ni tools")
    return respuesta


def recuperar_seguro(
    retriever: _Recuperador, consulta: str | None, trace_id: str = ""
) -> list[ResultadoRecuperacion] | RespuestaFalloSeguro:
    """Recupera fragmentos; un fallo específico del retriever -> fallo seguro."""
    try:
        return retriever.recuperar(consulta)
    except _ERRORES_RETRIEVER as exc:
        return fallar("retriever_error", trace_id, exc)


def revisar_resultado_tool(
    resultado: Any, trace_id: str = ""
) -> RespuestaFalloSeguro | None:
    """None si el resultado es usable; fallo seguro si no es dict o es `error_interno`."""
    if not isinstance(resultado, dict):
        return fallar(
            "tool_error_interno", trace_id,
            detalle=f"resultado de tool no es dict (tipo {type(resultado).__name__})",
        )
    if resultado.get("error") == "error_interno":
        return fallar("tool_error_interno", trace_id, detalle=str(resultado.get("mensaje", "")))
    return None
