"""Modelos de dominio: respuesta del agente, fragmentos, pedidos y respuesta del LLM."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Accion = Literal["responder", "escalar", "pedir_dato"]


class AgentResponse(BaseModel):
    """Resultado final del agente ante un mensaje del cliente."""

    model_config = ConfigDict(extra="forbid")

    respuesta: str
    accion: Accion
    fuentes: list[str] = Field(default_factory=list)
    canal: str | None = None
    trace_id: str


class Chunk(BaseModel):
    """Fragmento de un documento indexable."""

    model_config = ConfigDict(extra="forbid")

    texto: str
    doc_id: str
    metadatos: dict[str, Any] = Field(default_factory=dict)


def clave_chunk(chunk: Chunk) -> tuple[str, int]:
    """Clave estable `(doc_id, posicion)` de un chunk (la `posicion` la pone el loader)."""
    posicion = chunk.metadatos.get("posicion")
    if not isinstance(posicion, int) or isinstance(posicion, bool):
        raise ValueError(  # noqa: TRY004 - dato ausente o inválido: es un error de valor
            f"El chunk de «{chunk.doc_id}» no tiene metadatos['posicion'] entera "
            f"(valor: {posicion!r}); no se puede identificar de forma estable."
        )
    return (chunk.doc_id, posicion)


class Order(BaseModel):
    """Pedido de la tabla de pedidos (inmutable)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str
    producto: str
    estado: str
    entrega_estimada: str | None = None


class LlamadaTool(BaseModel):
    """Llamada a una tool solicitada por el LLM."""

    model_config = ConfigDict(extra="forbid")

    id: str
    nombre: str
    argumentos: dict[str, Any] = Field(default_factory=dict)


class UsoTokens(BaseModel):
    """Consumo de tokens de una llamada al LLM (para el tracing)."""

    model_config = ConfigDict(extra="forbid")

    entrada: int = Field(default=0, ge=0)
    salida: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    """Respuesta tipada del LLM: texto y/o llamadas a tools, más el uso de tokens."""

    model_config = ConfigDict(extra="forbid")

    texto: str | None = None
    llamadas_tools: list[LlamadaTool] = Field(default_factory=list)
    uso: UsoTokens = Field(default_factory=UsoTokens)
