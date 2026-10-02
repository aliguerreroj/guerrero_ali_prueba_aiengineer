"""Modelos de dominio: respuesta del agente, fragmentos de documentos y pedidos."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Accion = Literal["responder", "escalar", "pedir_dato"]


class AgentResponse(BaseModel):
    """Resultado final del agente ante un mensaje del cliente."""

    respuesta: str
    accion: Accion
    fuentes: list[str] = Field(default_factory=list)
    canal: str | None = None
    trace_id: str


class Chunk(BaseModel):
    """Fragmento de un documento indexable."""

    texto: str
    doc_id: str
    metadatos: dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    """Pedido de la tabla de pedidos."""

    order_id: str
    producto: str
    estado: str
    entrega_estimada: str | None = None
