"""API HTTP del agente (T20): FastAPI con `POST /chat` y `GET /health`.

Arranque (con el entorno activado):

    python -m uvicorn tiendahogar_agent.api:app --port 8000

La documentación interactiva queda en `/docs`. Ver `docs/api.md`.

Decisiones del prototipo:
- El orquestador se construye con `cli.construir_orquestador` (misma configuración que la CLI) la
  primera vez que se usa, no al importar el módulo.
- El historial vive en memoria (`AlmacenHistorial`): acotado en turnos y en número de
  conversaciones (se descarta la más antigua) y protegido con un lock. En producción iría en un
  almacén externo (p. ej. Redis, con TTL), porque la memoria del proceso no se comparte entre
  réplicas ni sobrevive a un reinicio.
- Sin autenticación: en producción la autenticación y el control de acceso (zero-trust) van en la
  capa de gestión de APIs (Apigee), delante de este servicio.
- Fallo seguro: si el dominio ya responde «escalar», se devuelve tal cual; ante un error
  imprevisto el cliente solo ve un 500 genérico (sin traza ni datos personales; el detalle va al log).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tiendahogar_agent.cli import construir_orquestador
from tiendahogar_agent.config import Settings, cargar_settings
from tiendahogar_agent.models import AgentResponse
from tiendahogar_agent.orquestador import MAX_MENSAJES_HISTORIAL

logger = logging.getLogger(__name__)

MAX_LARGO_MENSAJE = 2000
MAX_LARGO_CONVERSATION_ID = 64
PATRON_CONVERSATION_ID = r"^[A-Za-z0-9_.:-]+$"
MAX_CONVERSACIONES = 1000
MENSAJE_ERROR_INTERNO = "Error interno. Inténtalo de nuevo más tarde."


class ChatRequest(BaseModel):
    """Cuerpo de `POST /chat`."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(
        min_length=1, max_length=MAX_LARGO_CONVERSATION_ID, pattern=PATRON_CONVERSATION_ID
    )
    mensaje: str = Field(max_length=MAX_LARGO_MENSAJE)

    @field_validator("mensaje")
    @classmethod
    def _no_vacio(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("el mensaje no puede estar vacío")
        return valor


class AlmacenHistorial:
    """Historial por conversación en memoria (prototipo; en producción, Redis u otro externo).

    Acotado: máximo `max_mensajes` por conversación y `max_conversaciones` en total (al superarlo
    se descarta la menos recientemente usada). Seguro entre hilos.
    """

    def __init__(
        self,
        max_conversaciones: int = MAX_CONVERSACIONES,
        max_mensajes: int = MAX_MENSAJES_HISTORIAL,
    ) -> None:
        self._max_conversaciones = max_conversaciones
        self._max_mensajes = max_mensajes
        self._datos: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def obtener(self, conversation_id: str) -> list[dict[str, str]]:
        """Copia del historial (vacío si no existe); marca la conversación como reciente."""
        with self._lock:
            if conversation_id not in self._datos:
                return []
            self._datos.move_to_end(conversation_id)
            return [dict(m) for m in self._datos[conversation_id]]

    def agregar_turno(self, conversation_id: str, usuario: str, asistente: str) -> None:
        with self._lock:
            historial = self._datos.pop(conversation_id, [])
            historial += [
                {"role": "user", "content": usuario},
                {"role": "assistant", "content": asistente},
            ]
            del historial[: -self._max_mensajes]
            while historial and historial[0]["role"] != "user":
                historial.pop(0)
            self._datos[conversation_id] = historial
            while len(self._datos) > self._max_conversaciones:
                self._datos.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._datos)


def crear_app(
    orquestador: Any | None = None,
    settings: Settings | None = None,
    almacen: AlmacenHistorial | None = None,
) -> FastAPI:
    """Fábrica de la app. `orquestador` permite inyectar uno de prueba (FakeLLM, sin red)."""
    app = FastAPI(title="TiendaHogar · agente de soporte", version="0.1.0")
    historial = almacen if almacen is not None else AlmacenHistorial()
    estado: dict[str, Any] = {"orquestador": orquestador}
    lock_construccion = threading.Lock()

    def obtener_orquestador() -> Any:
        with lock_construccion:
            if estado["orquestador"] is None:
                estado["orquestador"] = construir_orquestador(settings or cargar_settings())
            return estado["orquestador"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=AgentResponse)
    def chat(peticion: ChatRequest) -> AgentResponse:
        try:
            orq = obtener_orquestador()
            respuesta = orq.procesar(peticion.mensaje, historial.obtener(peticion.conversation_id))
            historial.agregar_turno(peticion.conversation_id, peticion.mensaje, respuesta.respuesta)
            return respuesta
        except Exception as exc:  # noqa: BLE001  borde HTTP: nada de trazas ni PII al cliente
            logger.error("error interno en /chat (%s)", type(exc).__name__)
            raise HTTPException(status_code=500, detail=MENSAJE_ERROR_INTERNO) from None

    return app


app = crear_app()
