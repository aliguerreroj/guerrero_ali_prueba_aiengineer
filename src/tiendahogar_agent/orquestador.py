"""Orquestador (T14): une guardrails, clasificador, bucle de tool use y verificación de salida.

Flujo de `Orquestador.procesar` (el sistema decide la acción; el LLM solo redacta):
1. Entrada vacía/no textual -> `pedir_dato` con plantilla, sin LLM.
2. Reglas deterministas (T08) y clasificador LLM (T15, solo agrega escalamientos).
3. Si algo escala: el LLM redacta el mensaje SIN tools; se verifica (T10) y, si el LLM falla o
   no cumple (p. ej. omite el canal), se usa una plantilla de respaldo por categoría.
4. Intención `fuera_de_alcance`: plantilla amable determinista (sin LLM, sin fuentes).
5. Bucle de tool use (máx. `Settings.max_iteraciones_llm` llamadas al LLM) con `buscar_politicas`,
   `consultar_estado_pedido` y `responder`. Cualquier fallo -> fallo seguro (escalar).
6. Regla de acción: base `responder`; `accion_sugerida` del LLM solo se acepta si lleva hacia el
   lado más seguro (responder < pedir_dato < escalar). Nunca anula una decisión del sistema.
7. `verificar_salida` corre siempre antes de entregar; si no pasa, se escala con su respuesta.

Decisiones:
- PII: el LLM recibe el texto original del cliente y el historial tal cual (el enmascarado no
  toca ids `ORD-####`, pero un nombre o dato legítimo podría alterarse y degradar la respuesta);
  todo lo que se registra en logs va enmascarado y nunca se loguea el mensaje sin enmascarar.
- Historial: se filtran las entradas que no sean `user`/`assistant` con texto (así un cliente no
  puede inyectar un `system` ni un `tool`), se conservan los últimos `MAX_MENSAJES_HISTORIAL` y
  el primero siempre es `user`. Un historial que no es lista se ignora. Se registra cuántas
  entradas se descartaron, nunca su contenido. Las cifras dichas por el cliente en turnos
  previos cuentan como sustento en `verificar_salida`.
- Borde externo: toda excepción inesperada (también un bug de programación o un
  `ErrorHistorialMensajes`) se registra con traceback enmascarado y escala: de cara al cliente
  el fallo seguro tiene prioridad (a diferencia del clasificador, que propaga los bugs).
- Ids de fuentes: `buscar_politicas` presenta cada fragmento con `id` igual a su `doc_id`, de modo
  que lo que cita el LLM coincide con lo que valida `verificar_salida`.
- `timeout_tool_s` no se aplica: ambas tools son síncronas y locales.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from tiendahogar_agent.clasificador import aplicar_clasificador
from tiendahogar_agent.config import Settings
from tiendahogar_agent.guardrail_input import (
    CANAL_ESCALAMIENTO,
    ResultadoGuardrail,
    evaluar_con_settings,
)
from tiendahogar_agent.guardrail_output import verificar_salida
from tiendahogar_agent.mensajes import (
    mensaje_desde_respuesta,
    mensaje_resultado_tool,
)
from tiendahogar_agent.models import AgentResponse, Chunk, LlamadaTool, LLMResponse
from tiendahogar_agent.pedidos import OrderRepositoryMock
from tiendahogar_agent.pii import enmascarar_pii
from tiendahogar_agent.prompts import (
    ACCIONES_SUGERIBLES,
    NOMBRE_TOOL_BUSCAR,
    NOMBRE_TOOL_PEDIDO,
    NOMBRE_TOOL_RESPONDER,
    cargar_prompt_escalamiento,
    cargar_prompt_sistema,
    construir_contexto_documentos,
    definicion_tool_buscar_politicas,
    definicion_tool_consultar_pedido,
    definicion_tool_responder,
)
from tiendahogar_agent.puertos import LLMClient, OrderRepository
from tiendahogar_agent.resiliencia import (
    RespuestaFalloSeguro,
    fallar,
    llamar_llm_seguro,
    recuperar_seguro,
    revisar_resultado_tool,
)
from tiendahogar_agent.retriever import Retriever
from tiendahogar_agent.texto import es_vacio_visible

logger = logging.getLogger(__name__)

MAX_MENSAJES_HISTORIAL = 20
_MAX_CARACTERES_LOG = 200
_ROLES_HISTORIAL = ("user", "assistant")

MENSAJE_PEDIR_DATO = (
    "Hola, con gusto te ayudo. Cuéntame qué necesitas: puedo resolver dudas sobre garantía, "
    "devoluciones, envíos y reembolsos, o consultar el estado de un pedido si me das su número."
)
MENSAJE_FUERA_DE_ALCANCE = (
    "Gracias por escribirnos. Ese tema se sale de lo que puedo resolver, pero con gusto te ayudo "
    "con dudas de garantía, devoluciones, envíos, reembolsos y canales de contacto, o con el "
    "estado de un pedido si me compartes su número. ¿Te ayudo con alguno de esos temas?"
)
_MENSAJE_ESCALAMIENTO_GENERICO = (
    "Gracias por contarme tu caso. Prefiero que lo atienda directamente una persona de nuestro "
    f"equipo, así que escríbele a {CANAL_ESCALAMIENTO} y con gusto te ayudarán."
)
# Respaldo si el LLM falla o su texto no pasa la verificación: sin cifras ni compromisos.
PLANTILLAS_ESCALAMIENTO = {
    "legal": (
        "Entiendo tu preocupación y quiero que te atiendan bien. Los temas legales los maneja "
        f"directamente nuestro equipo humano, así que escríbele a {CANAL_ESCALAMIENTO} y con "
        "gusto te atenderán."
    ),
    "queja_trato": (
        "Lamento mucho que hayas tenido esa experiencia, gracias por contármelo. Las quejas sobre "
        f"el trato de un empleado las revisa nuestro equipo humano; escríbele a {CANAL_ESCALAMIENTO} "
        "para que la atiendan como corresponde."
    ),
    "facturacion": (
        "Entiendo lo importante que es aclarar un tema de facturación. Estas disputas las atiende "
        f"nuestro equipo humano, así que escríbele a {CANAL_ESCALAMIENTO} y con gusto lo "
        "revisarán contigo."
    ),
    "reembolso_alto": (
        "Gracias por explicarme tu situación. Un reembolso de este monto necesita la revisión de "
        f"un supervisor humano, así que escríbele a {CANAL_ESCALAMIENTO} para que revisen tu caso."
    ),
    "manipulacion": (
        "Gracias por escribirnos. No puedo cambiar mis reglas de atención, pero si necesitas "
        f"ayuda con tu caso, nuestro equipo humano te atenderá en {CANAL_ESCALAMIENTO}."
    ),
}
_RECORDATORIO = (
    "Recuerda entregar tu respuesta final al cliente con la herramienta `responder`."
)

_ORDEN_SEGURIDAD = {"responder": 0, "pedir_dato": 1, "escalar": 2}
_CAMPOS_RESPONDER = {"respuesta", "fuentes", "accion_sugerida"}


def accion_mas_segura(base: str, sugerida: Any) -> str:
    """`sugerida` solo se acepta si es más segura que `base` (responder < pedir_dato < escalar)."""
    if isinstance(sugerida, str) and _ORDEN_SEGURIDAD.get(sugerida, -1) > _ORDEN_SEGURIDAD[base]:
        return sugerida
    return base


def _resumir(texto: str) -> str:
    """Texto enmascarado y acotado para logs."""
    return enmascarar_pii(texto)[:_MAX_CARACTERES_LOG]


@dataclass
class _Evidencia:
    """Lo recuperado en el turno: sustento para `verificar_salida`."""

    chunks: list[Chunk] = field(default_factory=list)
    pedidos: list[dict[str, Any]] = field(default_factory=list)
    _vistos: set[tuple[str, str]] = field(default_factory=set)

    def agregar_chunks(self, nuevos: list[Chunk]) -> None:
        for chunk in nuevos:
            clave = (chunk.doc_id, chunk.texto)
            if clave not in self._vistos:
                self._vistos.add(clave)
                self.chunks.append(chunk)

    @property
    def resultado_pedido(self) -> dict[str, Any] | None:
        return {"pedidos": list(self.pedidos)} if self.pedidos else None


class Orquestador:
    """Atiende un mensaje del cliente y devuelve un `AgentResponse` (nunca lanza)."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        settings: Settings,
        pedidos: OrderRepository | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._settings = settings
        self._pedidos: OrderRepository = pedidos if pedidos is not None else OrderRepositoryMock()
        self._tools = [
            definicion_tool_buscar_politicas(),
            definicion_tool_consultar_pedido(),
            definicion_tool_responder(),
        ]

    # ------------------------------------------------------------------ API
    def procesar(
        self, mensaje: str, historial: list[dict[str, Any]] | None = None
    ) -> AgentResponse:
        """Procesa un turno. `historial`: turnos previos `{"role": "user"|"assistant", "content"}`."""
        trace_id = str(uuid.uuid4())
        try:
            respuesta = self._procesar(mensaje, historial, trace_id)
        except Exception as exc:  # noqa: BLE001  borde externo: fallo seguro = escalar
            respuesta = fallar("error_inesperado", trace_id, exc).respuesta
        logger.info("turno finalizado trace_id=%s accion=%s", trace_id, respuesta.accion)
        return respuesta

    # ------------------------------------------------------------------ flujo
    def _procesar(self, mensaje: Any, historial: Any, trace_id: str) -> AgentResponse:
        if not isinstance(mensaje, str) or es_vacio_visible(mensaje):
            logger.info("turno con entrada vacia trace_id=%s accion=pedir_dato", trace_id)
            return AgentResponse(
                respuesta=MENSAJE_PEDIR_DATO, accion="pedir_dato", trace_id=trace_id
            )
        logger.info("turno recibido trace_id=%s mensaje=%s", trace_id, _resumir(mensaje))
        turnos = self._normalizar_historial(historial, trace_id)
        usuario = "\n".join([t["content"] for t in turnos if t["role"] == "user"] + [mensaje])

        decision = evaluar_con_settings(mensaje, self._settings)
        decision, intencion = aplicar_clasificador(
            mensaje, decision, self._llm, self._settings, trace_id
        )
        if decision.escalar:
            return self._escalar(mensaje, turnos, usuario, decision, trace_id)
        if intencion is not None and intencion.intencion == "fuera_de_alcance":
            logger.info("fuera de alcance trace_id=%s", trace_id)
            return AgentResponse(
                respuesta=MENSAJE_FUERA_DE_ALCANCE, accion="responder", trace_id=trace_id
            )
        return self._bucle(mensaje, turnos, usuario, trace_id)

    @staticmethod
    def _normalizar_historial(historial: Any, trace_id: str) -> list[dict[str, str]]:
        if historial is None:
            return []
        if not isinstance(historial, (list, tuple)):
            logger.warning("historial ignorado (no es una lista) trace_id=%s", trace_id)
            return []
        validos = [
            {"role": t["role"], "content": t["content"]}
            for t in historial
            if isinstance(t, dict)
            and t.get("role") in _ROLES_HISTORIAL
            and isinstance(t.get("content"), str)
            and not es_vacio_visible(t["content"])
        ]
        descartados = len(historial) - len(validos)
        if descartados:
            logger.warning("historial: %d entrada(s) descartada(s) trace_id=%s", descartados, trace_id)
        validos = validos[-MAX_MENSAJES_HISTORIAL:]
        while validos and validos[0]["role"] != "user":
            validos.pop(0)
        return validos

    def _escalar(
        self,
        mensaje: str,
        turnos: list[dict[str, str]],
        usuario: str,
        decision: ResultadoGuardrail,
        trace_id: str,
    ) -> AgentResponse:
        logger.info(
            "escalamiento trace_id=%s categoria=%s regla=%s",
            trace_id, decision.categoria, decision.regla,
        )
        mensajes = [
            {"role": "system", "content": cargar_prompt_escalamiento(decision.categoria)},
            *turnos,
            {"role": "user", "content": mensaje},
        ]
        respuesta = llamar_llm_seguro(self._llm, mensajes, self._settings, trace_id=trace_id)
        texto: str | None = None
        if isinstance(respuesta, LLMResponse) and not es_vacio_visible(respuesta.texto):
            candidato = (respuesta.texto or "").strip()
            veredicto = verificar_salida(candidato, "escalar", [], [], None, usuario)
            if veredicto.ok:
                texto = candidato
            else:
                logger.warning(
                    "mensaje de escalamiento rechazado trace_id=%s reglas=%s",
                    trace_id, ",".join(veredicto.reglas_fallidas),
                )
        if texto is None:
            texto = PLANTILLAS_ESCALAMIENTO.get(decision.categoria, _MENSAJE_ESCALAMIENTO_GENERICO)
        return AgentResponse(
            respuesta=texto, accion="escalar", canal=CANAL_ESCALAMIENTO, trace_id=trace_id
        )

    # ------------------------------------------------------------------ bucle de tool use
    def _bucle(
        self, mensaje: str, turnos: list[dict[str, str]], usuario: str, trace_id: str
    ) -> AgentResponse:
        mensajes: list[dict[str, Any]] = [
            {"role": "system", "content": cargar_prompt_sistema()},
            *turnos,
            {"role": "user", "content": mensaje},
        ]
        evidencia = _Evidencia()
        recordado = False
        for _ in range(self._settings.max_iteraciones_llm):
            respuesta = llamar_llm_seguro(
                self._llm, mensajes, self._settings, tools=self._tools, trace_id=trace_id
            )
            if isinstance(respuesta, RespuestaFalloSeguro):
                return respuesta.respuesta
            # llamar_llm_seguro garantiza texto o tools: nunca se reinyecta un asistente vacío.
            mensajes.append(mensaje_desde_respuesta(respuesta))
            llamadas = respuesta.llamadas_tools
            if not llamadas:
                if recordado:
                    return fallar(
                        "bucle_sin_respuesta", trace_id, detalle="texto sin usar responder"
                    ).respuesta
                recordado = True
                mensajes.append({"role": "user", "content": _RECORDATORIO})
                continue
            finales = [ll for ll in llamadas if ll.nombre == NOMBRE_TOOL_RESPONDER]
            if len(finales) > 1:
                return fallar(
                    "bucle_sin_respuesta", trace_id, detalle="más de una llamada a responder"
                ).respuesta
            if finales and _error_argumentos_responder(finales[0].argumentos) is None:
                return self._entregar(finales[0].argumentos, evidencia, usuario, trace_id)
            for llamada in llamadas:
                resultado = self._ejecutar(llamada, evidencia, trace_id)
                if isinstance(resultado, AgentResponse):
                    return resultado
                mensajes.append(resultado)
        return fallar(
            "bucle_sin_respuesta", trace_id, detalle="iteraciones agotadas sin responder"
        ).respuesta

    def _ejecutar(
        self, llamada: LlamadaTool, evidencia: _Evidencia, trace_id: str
    ) -> dict[str, Any] | AgentResponse:
        """Ejecuta una tool: mensaje de resultado para el LLM, o la respuesta de fallo seguro."""
        args = llamada.argumentos
        if llamada.nombre == NOMBRE_TOOL_BUSCAR and _args_texto(args, "consulta"):
            encontrados = recuperar_seguro(self._retriever, args["consulta"], trace_id)
            if isinstance(encontrados, RespuestaFalloSeguro):
                return encontrados.respuesta
            evidencia.agregar_chunks([r.chunk for r in encontrados])
            contexto = construir_contexto_documentos(
                encontrados, ids=[r.fuente.doc_id for r in encontrados]
            )
            return _resultado(llamada, {"fragmentos": len(encontrados), "contexto": contexto})
        if llamada.nombre == NOMBRE_TOOL_PEDIDO and _args_texto(args, "order_id"):
            pedido = self._pedidos.consultar_estado_pedido(args["order_id"])
            fallo = revisar_resultado_tool(pedido, trace_id)
            if fallo is not None:
                return fallo.respuesta
            evidencia.pedidos.append(pedido)
            # Pedido inexistente o formato inválido NO escalan: el LLM lo explica o pide el dato.
            return _resultado(llamada, pedido, es_error="error" in pedido)
        if llamada.nombre in (NOMBRE_TOOL_BUSCAR, NOMBRE_TOOL_PEDIDO, NOMBRE_TOOL_RESPONDER):
            return _resultado(
                llamada, {"error": f"argumentos inválidos para {llamada.nombre}"}, es_error=True
            )
        return _resultado(
            llamada, {"error": f"herramienta desconocida: {_resumir(llamada.nombre)}"}, es_error=True
        )

    # ------------------------------------------------------------------ entrega
    def _entregar(
        self, args: dict[str, Any], evidencia: _Evidencia, usuario: str, trace_id: str
    ) -> AgentResponse:
        sugerida = args.get("accion_sugerida")
        if sugerida is not None and sugerida not in ACCIONES_SUGERIBLES:
            logger.info(
                "accion_sugerida ignorada (valor invalido) trace_id=%s tipo=%s",
                trace_id, type(sugerida).__name__,
            )
        accion = accion_mas_segura("responder", sugerida)
        fuentes = list(dict.fromkeys(args.get("fuentes") or []))
        veredicto = verificar_salida(
            args["respuesta"].strip(), accion,  # type: ignore[arg-type]  # accion es una Accion válida
            fuentes, evidencia.chunks, evidencia.resultado_pedido, usuario,
        )
        if not veredicto.ok:
            logger.warning(
                "verificacion de salida fallida trace_id=%s reglas=%s",
                trace_id, ",".join(veredicto.reglas_fallidas),
            )
        return AgentResponse(
            respuesta=veredicto.respuesta,
            accion=veredicto.accion,
            fuentes=veredicto.fuentes,
            canal=veredicto.canal,
            trace_id=trace_id,
        )


# ---------------------------------------------------------------------- utilidades
def _args_texto(args: dict[str, Any], campo: str) -> bool:
    """True si `args` es exactamente `{campo: str no vacío}`."""
    valor = args.get(campo)
    return set(args) == {campo} and isinstance(valor, str) and not es_vacio_visible(valor)


def _error_argumentos_responder(args: dict[str, Any]) -> str | None:
    """None si los argumentos de `responder` son válidos; si no, el motivo."""
    if not set(args) <= _CAMPOS_RESPONDER:
        return "campos desconocidos"
    texto = args.get("respuesta")
    if not isinstance(texto, str) or es_vacio_visible(texto):
        return "respuesta vacía o no textual"
    fuentes = args.get("fuentes")
    if fuentes is not None and not (
        isinstance(fuentes, list) and all(isinstance(f, str) for f in fuentes)
    ):
        return "fuentes debe ser una lista de textos"
    return None


def _resultado(llamada: LlamadaTool, contenido: dict[str, Any], es_error: bool = False) -> dict[str, Any]:
    return mensaje_resultado_tool(
        llamada.id, json.dumps(contenido, ensure_ascii=False), es_error=es_error
    )
