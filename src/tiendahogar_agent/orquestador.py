"""Orquestador (T14): une guardrails, clasificador, bucle de tool use y verificación de salida.

Flujo de `Orquestador.procesar` (el sistema decide la acción; el LLM solo redacta):
1. Entrada vacía/no textual -> `pedir_dato` con plantilla, sin LLM.
2. Reglas deterministas (T08) y clasificador LLM (T15, solo agrega escalamientos).
3. Si algo escala: el LLM redacta el mensaje SIN tools; se verifica (T10) y, si el LLM falla o
   no cumple (p. ej. omite el canal), se usa una plantilla de respaldo por categoría.
4. Intención `fuera_de_alcance`: plantilla amable determinista (sin LLM, sin fuentes).
5. RAG determinista: el retriever corre en CADA turno y los documentos recuperados se inyectan
   como mensaje de contexto (ids = `doc_id`); no depende de que el LLM llame a `buscar_politicas`.
6. Bucle de tool use (máx. `Settings.max_iteraciones_llm` llamadas al LLM) con `buscar_politicas`
   (auxiliar), `consultar_estado_pedido` y `responder`, con `tool_choice="any"` (el modelo debe
   llamar una tool). Falla real (LLM/retriever/tool/excepción) -> fallo seguro (escalar).
   `MENSAJE_FUERA_DE_ALCANCE` (acción `responder`) solo si el LLM dio texto suelto tras el
   recordatorio SIN haber intentado ninguna tool, sin documentos ni pedido en el turno y sin un
   id de pedido en el mensaje. Iteraciones agotadas o tool-calls inválidos son una falla real.
7. Regla de acción: base `responder`; `accion_sugerida` del LLM solo se acepta si lleva hacia el
   lado más seguro (responder < pedir_dato < escalar). Nunca anula una decisión del sistema.
   Además (ADR-008): si el texto remite a soporte@tiendahogar.example o cita doc5, la acción sube a
   `escalar`, salvo que el cliente pregunte por los canales de contacto (`pregunta_por_canal`).
8. `verificar_salida` corre siempre antes de entregar; si no pasa, se escala con su respuesta.

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
- Reconsulta por turno (ADR-007): antes del LLM se extraen los ids `ORD-####` (regex estricto) del
  mensaje y del historial (user y assistant, máx. 3 los más recientes) y se reconsultan en el
  repositorio; el resultado entra en la evidencia y en un mensaje de contexto. Del historial solo
  se usan ids, jamás lo que dijo el asistente. Al entregar, si se cita «pedidos» sin ningún pedido
  válido en la evidencia (habiendo consultas, todas con error), la cita se retira (T10 no cambia); si en el turno solo hubo ids
  inexistentes/inválidos y no se cita nada, la acción pasa a `pedir_dato`.
- Evidencia del turno: `verificar_salida` valida contra los chunks recuperados EN ESTE TURNO (el
  retriever automático más los de `buscar_politicas`, sin duplicar) y los pedidos consultados; la
  fuente «pedidos» solo es válida si la tool devolvió un pedido real.
- Ids de fuentes: `buscar_politicas` presenta cada fragmento con `id` igual a su `doc_id`, de modo
  que lo que cita el LLM coincide con lo que valida `verificar_salida`.
- `timeout_tool_s` no se aplica: ambas tools son síncronas y locales.
"""

from __future__ import annotations

import json
import logging
import re
import time
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
from tiendahogar_agent.guardrail_output import (
    FUENTE_PEDIDOS,
    PREFIJO_DETALLE_HECHO,
    R_HECHO,
    verificar_salida,
)
from tiendahogar_agent.hechos import detectar_discrepancias
from tiendahogar_agent.mensajes import (
    mensaje_desde_respuesta,
    mensaje_resultado_tool,
)
from tiendahogar_agent.models import AgentResponse, Chunk, LlamadaTool, LLMResponse
from tiendahogar_agent.pedidos import OrderRepositoryMock, _normalizar
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
from tiendahogar_agent.puertos import LLMClient, OrderRepository, TraceSink
from tiendahogar_agent.resiliencia import (
    RespuestaFalloSeguro,
    fallar,
    llamar_llm_seguro,
    recuperar_seguro,
    revisar_resultado_tool,
)
from tiendahogar_agent.retriever import Retriever
from tiendahogar_agent.texto import es_vacio_visible, quitar_tildes
from tiendahogar_agent.tracing import (
    LLMContado,
    construir_traza,
    marcar_respaldo,
    traza_actual,
    traza_de_turno,
)

logger = logging.getLogger(__name__)

MAX_MENSAJES_HISTORIAL = 20
_MAX_CARACTERES_LOG = 200
_ROLES_HISTORIAL = ("user", "assistant")
_PATRON_PEDIDO = re.compile(r"ord\s*-?\s*\d", re.IGNORECASE)
# Extracción ESTRICTA de ids para la reconsulta por turno (no usar el patrón laxo de arriba).
_PATRON_ID_ESTRICTO = re.compile(r"(?<![A-Za-z0-9])ORD-[0-9]{4}(?![0-9])", re.IGNORECASE | re.ASCII)
MAX_IDS_RECONSULTA = 3
_ERRORES_DE_DATO = ("no_encontrado", "formato_invalido")
FUENTE_CANALES = "doc5"
# Excepción de ADR-008: el cliente PREGUNTA por los canales/medios de contacto (texto sin tildes).
# Acotada a propósito; ante la duda no aplica y la respuesta que remite a soporte escala.
_PREGUNTA_CANAL = re.compile(
    r"\bcanal(?:es)?\b|\bcontact(?:o|os|ar|arlos|arnos|arlo|arte)\b|\bcomunicarme\b|\btelefono\b"
    r"|\bcorreo\s+de\s+soporte\b|\bhorario\s+de\s+atencion\b"
    r"|\b(?:donde|como)\s+(?:puedo\s+)?(?:escribo|escribir|reporto|reportar)\b"
    r"|\bcomo\s+(?:puedo\s+)?hablo\s+con\s+soporte\b"
)

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

# Obliga al modelo a llamar siempre una tool (si no, a veces responde con texto suelto).
TOOL_CHOICE_BUCLE = "any"

_ORDEN_SEGURIDAD = {"responder": 0, "pedir_dato": 1, "escalar": 2}
_CAMPOS_RESPONDER = {"respuesta", "fuentes", "accion_sugerida"}


def accion_mas_segura(base: str, sugerida: Any) -> str:
    """`sugerida` solo se acepta si es más segura que `base` (responder < pedir_dato < escalar)."""
    if isinstance(sugerida, str) and _ORDEN_SEGURIDAD.get(sugerida, -1) > _ORDEN_SEGURIDAD[base]:
        return sugerida
    return base


def pregunta_por_canal(mensaje: str) -> bool:
    """¿El cliente pregunta por los canales/medios de contacto? (heurística acotada, ADR-008)."""
    return bool(_PREGUNTA_CANAL.search(quitar_tildes(mensaje)))


def _remite_al_canal_humano(texto: str, fuentes: list[str]) -> bool:
    """El texto nombra el canal humano (sin tildes ni mayúsculas) o la respuesta cita doc5."""
    return CANAL_ESCALAMIENTO in quitar_tildes(texto) or FUENTE_CANALES in fuentes


def _resumir(texto: str) -> str:
    """Texto enmascarado y acotado para logs."""
    return enmascarar_pii(texto)[:_MAX_CARACTERES_LOG]


@dataclass
class _Evidencia:
    """Lo recuperado en el turno: sustento para `verificar_salida`."""

    chunks: list[Chunk] = field(default_factory=list)
    pedidos: list[dict[str, Any]] = field(default_factory=list)
    # Subconjunto de `pedidos`: consultas del mensaje actual o hechas por el LLM en este turno
    # (no las reconsultadas solo por el historial). Sirve para decidir `pedir_dato`.
    pedidos_turno: list[dict[str, Any]] = field(default_factory=list)
    _vistos: set[tuple[str, str]] = field(default_factory=set)

    def agregar_chunks(self, nuevos: list[Chunk]) -> None:
        for chunk in nuevos:
            clave = (chunk.doc_id, chunk.texto)
            if clave not in self._vistos:
                self._vistos.add(clave)
                self.chunks.append(chunk)

    @property
    def hay_pedido_valido(self) -> bool:
        return any("error" not in p for p in self.pedidos)

    @property
    def solo_errores_de_dato_en_turno(self) -> bool:
        """True si en este turno solo hubo consultas de pedido con id inexistente o inválido."""
        return bool(self.pedidos_turno) and all(
            p.get("error") in _ERRORES_DE_DATO for p in self.pedidos_turno
        )

    @property
    def resultado_pedido(self) -> dict[str, Any] | None:
        return {"pedidos": list(self.pedidos)} if self.pedidos else None


@dataclass(frozen=True)
class _PedirCorreccion:
    """La respuesta contradice un hecho de los documentos: se pide UNA corrección al LLM."""

    detalles: tuple[str, ...]


class Orquestador:
    """Atiende un mensaje del cliente y devuelve un `AgentResponse` (nunca lanza)."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        settings: Settings,
        pedidos: OrderRepository | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._llm = llm
        # Sin sink no se traza ni se escribe nada (T17); un fallo del sink nunca altera el turno.
        self._trace_sink = trace_sink
        self._llm_contado = LLMContado(llm)
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
        inicio = time.perf_counter()
        with traza_de_turno(self._trace_sink is not None) as traza:
            try:
                respuesta = self._procesar(mensaje, historial, trace_id)
            except Exception as exc:  # noqa: BLE001  borde externo: fallo seguro = escalar
                respuesta = fallar("error_inesperado", trace_id, exc).respuesta
            logger.info("turno finalizado trace_id=%s accion=%s", trace_id, respuesta.accion)
            if traza is not None and self._trace_sink is not None:
                try:
                    latencia_ms = (time.perf_counter() - inicio) * 1000
                    self._trace_sink.registrar(
                        construir_traza(respuesta, traza, self._settings, latencia_ms)
                    )
                except Exception as exc:  # noqa: BLE001  la traza nunca rompe el turno
                    logger.warning(
                        "no se pudo registrar la traza trace_id=%s tipo=%s",
                        trace_id, type(exc).__name__,
                    )
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
            mensaje, decision, self._llm_contado, self._settings, trace_id
        )
        traza = traza_actual()
        if traza is not None and decision.escalar:
            traza.categoria, traza.regla = decision.categoria, decision.regla
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
        respuesta = llamar_llm_seguro(self._llm_contado, mensajes, self._settings, trace_id=trace_id)
        texto: str | None = None
        if isinstance(respuesta, LLMResponse) and not es_vacio_visible(respuesta.texto):
            candidato = (respuesta.texto or "").strip()
            veredicto = verificar_salida(candidato, "escalar", [], [], None, usuario)
            _registrar_reglas_fallidas(veredicto.reglas_fallidas)
            if veredicto.ok:
                texto = candidato
            else:
                logger.warning(
                    "mensaje de escalamiento rechazado trace_id=%s reglas=%s",
                    trace_id, ",".join(veredicto.reglas_fallidas),
                )
        if texto is None:
            marcar_respaldo()
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
        encontrados = recuperar_seguro(self._retriever, mensaje, trace_id)
        if isinstance(encontrados, RespuestaFalloSeguro):
            return encontrados.respuesta
        evidencia.agregar_chunks([r.chunk for r in encontrados])
        _registrar_documentos(encontrados, "rag_automatico")
        logger.info("rag por turno trace_id=%s fragmentos=%d", trace_id, len(encontrados))
        mensajes.insert(
            1,
            {
                "role": "system",
                "content": construir_contexto_documentos(
                    encontrados, ids=[r.fuente.doc_id for r in encontrados]
                ),
            },
        )
        reconsulta = self._reconsultar_pedidos(mensaje, turnos, evidencia, trace_id)
        if isinstance(reconsulta, RespuestaFalloSeguro):
            return reconsulta.respuesta
        if reconsulta:
            mensajes.insert(2, {"role": "system", "content": reconsulta})
        recordado = False
        intento_tools = False
        # ADR-009: un solo reintento por turno ante `hecho_incorrecto`; no consume el tope de
        # iteraciones (lo amplía en uno), de modo que nunca hay más de una llamada extra.
        reintento_hecho_usado = False
        limite = self._settings.max_iteraciones_llm
        usadas = 0
        while usadas < limite:
            usadas += 1
            respuesta = llamar_llm_seguro(
                self._llm_contado, mensajes, self._settings, tools=self._tools, trace_id=trace_id,
                tool_choice=TOOL_CHOICE_BUCLE,
            )
            if isinstance(respuesta, RespuestaFalloSeguro):
                return respuesta.respuesta
            # llamar_llm_seguro garantiza texto o tools: nunca se reinyecta un asistente vacío.
            mensajes.append(mensaje_desde_respuesta(respuesta))
            llamadas = respuesta.llamadas_tools
            if not llamadas:
                if recordado:
                    return self._texto_suelto(
                        mensaje, evidencia, intento_tools, trace_id
                    )
                recordado = True
                mensajes.append({"role": "user", "content": _RECORDATORIO})
                continue
            finales = [ll for ll in llamadas if ll.nombre == NOMBRE_TOOL_RESPONDER]
            if len(finales) > 1:
                return fallar(
                    "bucle_sin_respuesta", trace_id, detalle="más de una llamada a responder"
                ).respuesta
            if finales and _error_argumentos_responder(finales[0].argumentos) is None:
                _registrar_tool(
                    NOMBRE_TOOL_RESPONDER,
                    {k: v for k, v in finales[0].argumentos.items() if k != "respuesta"},
                )
                entrega = self._entregar(
                    finales[0].argumentos, evidencia, usuario, trace_id, mensaje,
                    reintento_hecho_usado=reintento_hecho_usado,
                )
                if isinstance(entrega, AgentResponse):
                    return entrega
                reintento_hecho_usado = True
                limite += 1
                mensajes.append(_resultado(
                    finales[0],
                    {
                        "error": "hecho_incorrecto",
                        "instruccion": (
                            "Tu respuesta contradice los documentos. Corrígela con el hecho "
                            "correcto y su fuente (cita su doc_id en fuentes) y vuelve a llamar "
                            "a responder."
                        ),
                        "hecho_correcto": entrega.detalles,
                    },
                    es_error=True,
                ))
                # Historial válido en proveedores reales: toda tool_call necesita su resultado.
                mensajes.extend(
                    _resultado(
                        ll,
                        {"error": "no ejecutada: corrige primero tu respuesta y vuelve a llamar a responder"},
                        es_error=True,
                    )
                    for ll in llamadas
                    if ll is not finales[0]
                )
                continue
            intento_tools = True
            for llamada in llamadas:
                resultado = self._ejecutar(llamada, evidencia, trace_id)
                if isinstance(resultado, AgentResponse):
                    return resultado
                mensajes.append(resultado)
        return fallar(
            "bucle_sin_respuesta", trace_id, detalle="iteraciones agotadas sin responder"
        ).respuesta

    def _reconsultar_pedidos(
        self, mensaje: str, turnos: list[dict[str, str]], evidencia: _Evidencia, trace_id: str
    ) -> str | RespuestaFalloSeguro | None:
        """Reconsulta determinista (ADR-007): ids del mensaje y del historial -> repositorio.

        El repositorio es la única fuente de verdad: del historial solo se toman los ids, nunca
        lo que dijo el asistente. Devuelve el mensaje de contexto para el LLM (o None si no hay
        ids) o el fallo seguro si el repositorio falla.
        """
        ids_actuales = extraer_ids_pedido([mensaje], MAX_IDS_RECONSULTA)
        ids = extraer_ids_pedido([t["content"] for t in turnos] + [mensaje], MAX_IDS_RECONSULTA)
        if not ids:
            return None
        bloques: list[str] = []
        for order_id in ids:
            pedido = self._pedidos.consultar_estado_pedido(order_id)
            fallo = revisar_resultado_tool(pedido, trace_id)
            if fallo is not None:
                return fallo
            evidencia.pedidos.append(pedido)
            _registrar_tool(NOMBRE_TOOL_PEDIDO, {"order_id": order_id}, "reconsulta_automatica")
            if order_id in ids_actuales:
                evidencia.pedidos_turno.append(pedido)
            bloques.append(json.dumps(pedido, ensure_ascii=False))
        logger.info("reconsulta de pedidos trace_id=%s cantidad=%d", trace_id, len(ids))
        return (
            "Consulta de pedido ya realizada por el sistema en este turno (datos autoritativos de "
            "la tabla de pedidos; no hace falta repetirla con la herramienta). Cita «pedidos» solo "
            "si usas un pedido que aparezca aquí sin error:\n" + "\n".join(bloques)
        )

    @staticmethod
    def _texto_suelto(
        mensaje: str, evidencia: _Evidencia, intento_tools: bool, trace_id: str
    ) -> AgentResponse:
        """El LLM insistió en texto suelto tras el recordatorio.

        Solo es un tema ajeno (plantilla amable, `responder`) si no intentó ninguna tool, no hubo
        documentos ni pedido en el turno y el mensaje no menciona un pedido. En cualquier otro
        caso es una falla real: fallo seguro.
        """
        if (
            not intento_tools
            and not evidencia.chunks
            and not evidencia.pedidos
            and not _PATRON_PEDIDO.search(mensaje)
        ):
            logger.warning("texto suelto sin evidencia: fuera de alcance trace_id=%s", trace_id)
            marcar_respaldo()
            return AgentResponse(
                respuesta=MENSAJE_FUERA_DE_ALCANCE, accion="responder", trace_id=trace_id
            )
        return fallar("bucle_sin_respuesta", trace_id, detalle="texto sin usar responder").respuesta

    def _ejecutar(
        self, llamada: LlamadaTool, evidencia: _Evidencia, trace_id: str
    ) -> dict[str, Any] | AgentResponse:
        """Ejecuta una tool: mensaje de resultado para el LLM, o la respuesta de fallo seguro."""
        args = llamada.argumentos
        _registrar_tool(llamada.nombre, args)
        if llamada.nombre == NOMBRE_TOOL_BUSCAR and _args_texto(args, "consulta"):
            encontrados = recuperar_seguro(self._retriever, args["consulta"], trace_id)
            if isinstance(encontrados, RespuestaFalloSeguro):
                return encontrados.respuesta
            evidencia.agregar_chunks([r.chunk for r in encontrados])
            _registrar_documentos(encontrados, "buscar_politicas")
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
            evidencia.pedidos_turno.append(pedido)
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
        self,
        args: dict[str, Any],
        evidencia: _Evidencia,
        usuario: str,
        trace_id: str,
        mensaje: str = "",
        reintento_hecho_usado: bool = False,
    ) -> AgentResponse | _PedirCorreccion:
        sugerida = args.get("accion_sugerida")
        if sugerida is not None and sugerida not in ACCIONES_SUGERIBLES:
            logger.info(
                "accion_sugerida ignorada (valor invalido) trace_id=%s tipo=%s",
                trace_id, type(sugerida).__name__,
            )
        accion = accion_mas_segura("responder", sugerida)
        fuentes = list(dict.fromkeys(args.get("fuentes") or []))
        if FUENTE_PEDIDOS in fuentes and evidencia.pedidos and not evidencia.hay_pedido_valido:
            # Cita sin respaldo (se consultó y dio error): se retira, lo cual no inventa nada. T10
            # sigue verificando cifras y las demás fuentes.
            fuentes = [f for f in fuentes if f != FUENTE_PEDIDOS]
            logger.info("fuente pedidos retirada (sin pedido valido) trace_id=%s", trace_id)
        if _remite_al_canal_humano(args["respuesta"], fuentes) and not pregunta_por_canal(mensaje):
            # ADR-008: si el texto remite a soporte, la acción debe decirlo (solo sube, nunca baja).
            # Si no trae el canal, verificar_salida (R_CANAL) lo cambia por la respuesta segura.
            if accion != "escalar":
                logger.info("remision a soporte detectada: se escala trace_id=%s", trace_id)
            accion = accion_mas_segura(accion, "escalar")
        if not fuentes and evidencia.solo_errores_de_dato_en_turno:
            # Solo hubo ids inexistentes/inválidos y no se cita nada: la respuesta necesariamente
            # pide revisar el número. Lado seguro: pedir_dato (entre responder y escalar).
            accion = accion_mas_segura(accion, "pedir_dato")
        veredicto = verificar_salida(
            args["respuesta"].strip(), accion,  # type: ignore[arg-type]  # accion es una Accion válida
            fuentes, evidencia.chunks, evidencia.resultado_pedido, usuario,
        )
        _registrar_reglas_fallidas(veredicto.reglas_fallidas)
        if R_HECHO in veredicto.reglas_fallidas and not reintento_hecho_usado:
            # El hecho correcto sale de la tabla de hechos (texto literal de los documentos): se
            # añade como evidencia para que la corrección pueda citarlo y sustentar su cifra.
            discrepancias = detectar_discrepancias(args["respuesta"])
            for d in discrepancias:
                if d.hecho is not None:
                    evidencia.agregar_chunks([Chunk(
                        texto=d.hecho.frase_origen, doc_id=d.hecho.doc_id,
                        metadatos={"origen": "tabla_de_hechos"},
                    )])
            logger.warning(
                "hecho_incorrecto detectado, se reintenta una vez trace_id=%s hechos=%d",
                trace_id, len(discrepancias),
            )
            traza = traza_actual()
            if traza is not None:
                traza.reintentos = 1
            return _PedirCorreccion(tuple(
                d[len(PREFIJO_DETALLE_HECHO):] for d in veredicto.detalles
                if d.startswith(PREFIJO_DETALLE_HECHO)
            ))
        if reintento_hecho_usado and R_HECHO in veredicto.reglas_fallidas:
            logger.warning("reintento por hecho_incorrecto no resolvio trace_id=%s", trace_id)
        if not veredicto.ok:
            marcar_respaldo()
            logger.warning(
                "verificacion de salida fallida trace_id=%s reglas=%s",
                trace_id, ",".join(veredicto.reglas_fallidas),
            )
        elif reintento_hecho_usado:
            logger.warning("reintento por hecho_incorrecto resuelto trace_id=%s", trace_id)
        return AgentResponse(
            respuesta=veredicto.respuesta,
            accion=veredicto.accion,
            fuentes=veredicto.fuentes,
            canal=veredicto.canal,
            trace_id=trace_id,
        )


# ---------------------------------------------------------------------- utilidades
def _registrar_documentos(resultados: list[Any], origen: str) -> None:
    traza = traza_actual()
    if traza is not None:
        traza.agregar_documentos(resultados, origen)


def _registrar_tool(nombre: str, argumentos: Any, origen: str = "llm") -> None:
    traza = traza_actual()
    if traza is not None:
        traza.agregar_tool(nombre, argumentos, origen)


def _registrar_reglas_fallidas(reglas: Any) -> None:
    traza = traza_actual()
    if traza is not None:
        traza.reglas_fallidas.extend(str(r) for r in reglas)


def extraer_ids_pedido(textos: list[str], maximo: int = MAX_IDS_RECONSULTA) -> list[str]:
    """Ids `ORD-####` (regex estricto) normalizados, sin repetir; los `maximo` más recientes.

    Se recorre en orden de aparición; si un id se repite, cuenta su última aparición.
    """
    vistos: dict[str, None] = {}
    for texto in textos:
        for crudo in _PATRON_ID_ESTRICTO.findall(texto):
            normalizado = _normalizar(crudo)
            if normalizado is not None:
                vistos.pop(normalizado, None)
                vistos[normalizado] = None
    return list(vistos)[-maximo:] if maximo > 0 else []


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
