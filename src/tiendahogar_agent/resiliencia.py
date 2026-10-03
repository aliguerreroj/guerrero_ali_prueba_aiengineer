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
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from tiendahogar_agent.config import Settings
from tiendahogar_agent.excepciones import ErrorLLM, ErrorRecuperacion
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.models import AgentResponse, LLMResponse
from tiendahogar_agent.pii import enmascarar_pii
from tiendahogar_agent.retriever import ResultadoRecuperacion

logger = logging.getLogger(__name__)

MotivoFallo = Literal["llm_error", "llm_respuesta_vacia", "tool_error_interno", "retriever_error"]

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


def _fallar(
    motivo: MotivoFallo,
    trace_id: str,
    exc: BaseException | None = None,
    detalle: str = "",
) -> RespuestaFalloSeguro:
    """Registra el fallo (PII enmascarada, sin traceback) y devuelve el fallo seguro."""
    tipo = type(exc).__name__ if exc is not None else "-"
    mensaje = enmascarar_pii(str(exc) if exc is not None else detalle)
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
) -> LLMResponse | RespuestaFalloSeguro:
    """Llama al LLM con `settings.timeout_llm_s`; error o respuesta vacía -> fallo seguro."""
    try:
        respuesta = llm.completar(mensajes, tools=tools, timeout=settings.timeout_llm_s)
    except ErrorLLM as exc:
        return _fallar("llm_error", trace_id, exc)
    if not (respuesta.texto or "").strip() and not respuesta.llamadas_tools:
        return _fallar("llm_respuesta_vacia", trace_id, detalle="el LLM no devolvió texto ni tools")
    return respuesta


def recuperar_seguro(
    retriever: _Recuperador, consulta: str | None, trace_id: str = ""
) -> list[ResultadoRecuperacion] | RespuestaFalloSeguro:
    """Recupera fragmentos; un fallo específico del retriever -> fallo seguro."""
    try:
        return retriever.recuperar(consulta)
    except _ERRORES_RETRIEVER as exc:
        return _fallar("retriever_error", trace_id, exc)


def revisar_resultado_tool(
    resultado: dict[str, Any], trace_id: str = ""
) -> RespuestaFalloSeguro | None:
    """None si el resultado de la tool es usable; fallo seguro si es `error_interno`."""
    if isinstance(resultado, dict) and resultado.get("error") == "error_interno":
        return _fallar("tool_error_interno", trace_id, detalle=str(resultado.get("mensaje", "")))
    return None
