"""Clasificador de intención (T15): segunda capa, tras las reglas deterministas de T08.

Reglas de diseño:
- Salida estructurada por una tool del dominio (`clasificar_intencion`) con enum estricto;
  solo se confía en sus argumentos (nunca en texto libre) y se validan con pydantic.
- Solo puede AGREGAR escalamientos: `combinar` devuelve intacta la decisión de las reglas si
  estas escalaron (y `aplicar_clasificador` ni siquiera llama al LLM); la clasificación
  nunca produce «responder» que anule algo.
- El mensaje se enmascara (PII) y se delimita como datos antes de llegar al LLM.
- Cualquier fallo del clasificador (error del LLM, respuesta vacía, tool equivocada, args
  inválidos...) NO bloquea: devuelve «sin clasificación» y sigue la decisión de las reglas.
  Los logs no incluyen el mensaje del cliente.
- Aritmética, traducciones, código y similares se clasifican `fuera_de_alcance` (prompt
  `clasificador.md`, hallazgo 2, 2026-10-04): el orquestador responde con la plantilla
  determinista sin llamar al bucle, así el agente nunca resuelve ni da el resultado.
- Los bugs de programación (TypeError, AttributeError...) se propagan.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from tiendahogar_agent.config import Settings
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO, Categoria, ResultadoGuardrail
from tiendahogar_agent.pii import enmascarar_pii
from tiendahogar_agent.prompts import _leer_plantilla, _neutralizar
from tiendahogar_agent.resiliencia import RespuestaFalloSeguro, llamar_llm_seguro
from tiendahogar_agent.texto import es_vacio_visible

logger = logging.getLogger(__name__)

NOMBRE_TOOL = "clasificar_intencion"
REGLA_CLASIFICADOR = "clasificador_llm"

Intencion = Literal["politica", "pedido", "escalar", "fuera_de_alcance"]
CategoriaEscalable = Literal[
    "reembolso_alto", "queja_trato", "facturacion", "legal", "manipulacion"
]
# Si Categoria cambia, las categorías escalables deben seguirla (todas salvo "ninguna").
if set(get_args(CategoriaEscalable)) != set(get_args(Categoria)) - {"ninguna"}:  # pragma: no cover
    raise RuntimeError("CategoriaEscalable no coincide con Categoria del guardrail de entrada")

# Errores de programación: no se disfrazan de «fallo del clasificador».
_ERRORES_PROGRAMACION = (TypeError, AttributeError, NameError, AssertionError, NotImplementedError)


def _validar_coherencia(intencion: str | None, categoria: str | None) -> None:
    if intencion == "escalar" and categoria is None:
        raise ValueError("la intención 'escalar' exige categoria")
    if intencion != "escalar" and categoria is not None:
        raise ValueError("categoria solo se admite con la intención 'escalar'")


class _ArgumentosTool(BaseModel):
    """Argumentos de la tool, validados en modo estricto."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    intencion: Intencion
    categoria: CategoriaEscalable | None = None

    @model_validator(mode="after")
    def _coherente(self) -> _ArgumentosTool:
        _validar_coherencia(self.intencion, self.categoria)
        return self


class ResultadoIntencion(BaseModel):
    """Intención clasificada; `clasificado=False` significa «sin clasificación»."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intencion: Intencion | None = None
    categoria: CategoriaEscalable | None = None
    clasificado: bool = False

    @model_validator(mode="after")
    def _coherente(self) -> ResultadoIntencion:
        if self.clasificado != (self.intencion is not None):
            raise ValueError("clasificado debe ser True si y solo si hay intención")
        _validar_coherencia(self.intencion, self.categoria)
        return self


SIN_CLASIFICACION = ResultadoIntencion()


def definicion_tool_clasificar() -> dict[str, Any]:
    """Tool de salida estructurada en formato de dominio {name, description, parameters}."""
    return {
        "name": NOMBRE_TOOL,
        "description": "Registra la intención del mensaje del cliente. Llámala una sola vez.",
        "parameters": {
            "type": "object",
            "properties": {
                "intencion": {"type": "string", "enum": list(get_args(Intencion))},
                "categoria": {
                    "type": "string",
                    "enum": list(get_args(CategoriaEscalable)),
                    "description": "Obligatoria solo si intencion es escalar.",
                },
            },
            "required": ["intencion"],
            "additionalProperties": False,
        },
    }


def _fallo(motivo: str, trace_id: str, tipo: str = "-") -> ResultadoIntencion:
    # Solo motivo, tipo de error y trace_id: nunca el mensaje del cliente.
    logger.warning(
        "clasificador sin clasificacion motivo=%s tipo=%s trace_id=%s", motivo, tipo, trace_id
    )
    return SIN_CLASIFICACION


def _mensajes(mensaje: str) -> list[dict[str, Any]]:
    seguro = _neutralizar(enmascarar_pii(mensaje))
    return [
        {"role": "system", "content": _leer_plantilla("clasificador.md")},
        {"role": "user", "content": f"<mensaje_cliente>\n{seguro}\n</mensaje_cliente>"},
    ]


def clasificar_intencion(
    mensaje: str, llm: Any, settings: Settings, trace_id: str = ""
) -> ResultadoIntencion:
    """Clasifica la intención; ante cualquier fallo devuelve «sin clasificación»."""
    if not isinstance(mensaje, str) or es_vacio_visible(mensaje):
        return SIN_CLASIFICACION
    try:
        respuesta = llamar_llm_seguro(
            llm, _mensajes(mensaje), settings, tools=[definicion_tool_clasificar()],
            trace_id=trace_id, max_tokens=settings.max_tokens_clasificador,
        )
    except _ERRORES_PROGRAMACION:
        raise
    except Exception as exc:  # noqa: BLE001  # error inesperado del LLM: el clasificador es opcional
        return _fallo("llm_excepcion_inesperada", trace_id, type(exc).__name__)
    if isinstance(respuesta, RespuestaFalloSeguro):
        # No es un escalamiento: solo falló el clasificador (ya registrado por resiliencia).
        return SIN_CLASIFICACION
    llamadas = respuesta.llamadas_tools
    if len(llamadas) != 1:
        return _fallo("numero_de_tool_calls_invalido", trace_id)
    if llamadas[0].nombre != NOMBRE_TOOL:
        return _fallo("tool_equivocada", trace_id)
    try:
        args = _ArgumentosTool.model_validate(llamadas[0].argumentos)
    except ValidationError:
        return _fallo("argumentos_invalidos", trace_id)
    return ResultadoIntencion(intencion=args.intencion, categoria=args.categoria, clasificado=True)


def combinar(
    decision_reglas: ResultadoGuardrail, intencion: ResultadoIntencion | None
) -> ResultadoGuardrail:
    """Reglas primero; el clasificador solo puede agregar un escalamiento, nunca quitarlo."""
    if decision_reglas.escalar:
        return decision_reglas
    if intencion is None or not intencion.clasificado or intencion.intencion != "escalar":
        return decision_reglas
    return ResultadoGuardrail(
        escalar=True,
        categoria=intencion.categoria,  # type: ignore[arg-type]  # validada: nunca None aquí
        motivo=(
            "Se escala porque el clasificador LLM detectó un caso que debe atender una persona "
            f"({intencion.categoria})."
        ),
        regla=REGLA_CLASIFICADOR,
        accion="escalar",
        canal=CANAL_ESCALAMIENTO,
    )


def aplicar_clasificador(
    mensaje: str,
    decision_reglas: ResultadoGuardrail,
    llm: Any,
    settings: Settings,
    trace_id: str = "",
) -> tuple[ResultadoGuardrail, ResultadoIntencion | None]:
    """Segunda capa: (decisión final, intención o None si no se consultó al LLM)."""
    if not settings.usar_clasificador_llm or decision_reglas.escalar:
        return decision_reglas, None
    intencion = clasificar_intencion(mensaje, llm, settings, trace_id)
    return combinar(decision_reglas, intencion), intencion
