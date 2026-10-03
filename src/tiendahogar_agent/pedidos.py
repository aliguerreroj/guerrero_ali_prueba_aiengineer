"""Tool de consulta de pedidos y repositorio mock (implementa el puerto OrderRepository)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from tiendahogar_agent.models import Order

_FORMATO = re.compile(r"ORD-[0-9]{4}", re.ASCII)
# Tope de longitud previo a cualquier procesamiento (evita trabajo con entradas enormes).
_MAX_LONGITUD_ENTRADA = 64

PEDIDOS_MOCK: Mapping[str, Order] = MappingProxyType({
    p.order_id: p
    for p in (
        Order(order_id="ORD-1001", producto="Refrigeradora", estado="En tránsito",
              entrega_estimada="3 días hábiles"),
        Order(order_id="ORD-1002", producto="Licuadora", estado="Entregado",
              entrega_estimada=None),
        Order(order_id="ORD-1003", producto="Lavadora", estado="Procesando",
              entrega_estimada="6 días hábiles"),
        Order(order_id="ORD-1004", producto="Tostadora", estado="Cancelado",
              entrega_estimada=None),
    )
})


def _normalizar(order_id: object) -> str | None:
    """Devuelve el id normalizado (ORD-####) o None si el formato no es válido."""
    if not isinstance(order_id, str) or len(order_id) > _MAX_LONGITUD_ENTRADA:
        return None
    candidato = "".join(order_id.split()).upper()  # quita todo espacio y pasa a mayúsculas
    return candidato if _FORMATO.fullmatch(candidato) else None


class OrderRepositoryMock:
    """Repositorio de pedidos en memoria con la tabla mock."""

    def __init__(self, pedidos: Mapping[str, Order] | None = None) -> None:
        self._pedidos = dict(PEDIDOS_MOCK if pedidos is None else pedidos)

    def consultar_estado_pedido(self, order_id: str) -> dict[str, Any]:
        """Consulta el estado de un pedido. Nunca lanza excepción.

        Éxito (claves del modelo Order; entrega_estimada puede ser None):
            {"order_id": "ORD-1001", "producto": ..., "estado": ..., "entrega_estimada": ...}
        Error (sin claves de pedido; la clave "error" solo existe en errores):
            {"order_id": "ORD-9999", "error": "no_encontrado", "mensaje": ...}
            {"order_id": None, "error": "formato_invalido", "mensaje": ...}
            {"order_id": None, "error": "error_interno", "mensaje": ...}
        "error_interno" es un fallo inesperado de la propia tool (no un dato del
        cliente): el orquestador debe tratarlo como fallo seguro = escalar a un humano.
        Se usa la clave separada "error" (no "estado") para que un estado de pedido
        y un fallo de la consulta nunca se confundan. Normalización: se ignoran
        espacios (incluidos los internos, p. ej. "ORD - 1001") y mayúsculas/minúsculas;
        el formato debe ser ORD- seguido de exactamente 4 dígitos ASCII.
        """
        try:
            normalizado = _normalizar(order_id)
            if normalizado is None:
                return {
                    "order_id": None,
                    "error": "formato_invalido",
                    "mensaje": "El número de pedido debe tener el formato ORD-#### (por ejemplo, ORD-1001).",
                }
            pedido = self._pedidos.get(normalizado)
            if pedido is None:
                return {
                    "order_id": normalizado,
                    "error": "no_encontrado",
                    "mensaje": f"No encontré un pedido con el número {normalizado}.",
                }
            return pedido.model_dump()
        except Exception:  # noqa: BLE001  defensa final: la tool jamás propaga excepciones
            return {
                "order_id": None,
                "error": "error_interno",
                "mensaje": "No pude consultar el pedido por un problema interno.",
            }


_REPOSITORIO_POR_DEFECTO = OrderRepositoryMock()


def consultar_estado_pedido(order_id: str) -> dict:
    """Tool: consulta el estado de un pedido. Ver OrderRepositoryMock.consultar_estado_pedido
    para la forma exacta del dict (éxito, no_encontrado, formato_invalido,
    error_interno; este último debe escalarse a un humano)."""
    return _REPOSITORIO_POR_DEFECTO.consultar_estado_pedido(order_id)
