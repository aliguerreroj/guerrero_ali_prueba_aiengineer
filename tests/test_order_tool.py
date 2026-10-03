"""Pruebas de la tool consultar_estado_pedido y del repositorio mock."""

from __future__ import annotations

import inspect
import json

import pytest

from tiendahogar_agent.models import Order
from tiendahogar_agent.pedidos import OrderRepositoryMock, consultar_estado_pedido
from tiendahogar_agent.puertos import OrderRepository

ESPERADOS = {
    "ORD-1001": ("Refrigeradora", "En tránsito", "3 días hábiles"),
    "ORD-1002": ("Licuadora", "Entregado", None),
    "ORD-1003": ("Lavadora", "Procesando", "6 días hábiles"),
    "ORD-1004": ("Tostadora", "Cancelado", None),
}


@pytest.mark.parametrize("order_id", list(ESPERADOS))
def test_pedidos_existentes(order_id):
    producto, estado, entrega = ESPERADOS[order_id]
    r = consultar_estado_pedido(order_id)
    assert r == {
        "order_id": order_id,
        "producto": producto,
        "estado": estado,
        "entrega_estimada": entrega,
    }
    assert set(r) == set(Order.model_fields)
    assert "error" not in r


def test_entrega_nula_en_entregado_y_cancelado():
    assert consultar_estado_pedido("ORD-1002")["entrega_estimada"] is None
    assert consultar_estado_pedido("ORD-1004")["entrega_estimada"] is None


@pytest.mark.parametrize(
    "entrada",
    ["ord-1001", "  ORD-1001  ", "\tOrd-1001\n", "ORD - 1001", "o r d - 1 0 0 1", "oRd-1001"],
)
def test_normalizacion(entrada):
    r = consultar_estado_pedido(entrada)
    assert r["order_id"] == "ORD-1001"
    assert r["producto"] == "Refrigeradora"


def test_no_encontrado():
    r = consultar_estado_pedido(" ord-9999 ")
    assert r["error"] == "no_encontrado"
    assert r["order_id"] == "ORD-9999"
    assert "producto" not in r and "estado" not in r
    assert isinstance(r["mensaje"], str) and r["mensaje"]


@pytest.mark.parametrize(
    "entrada",
    ["ORD-99", "ORD-10001", "ORD1001", "PED-1001", "ORD-", "ORD-ABCD", "1001",
     "ORD-１００１", "ORD-1001; DROP", "ORD-1001\nORD-1002", "", "   ", "\x00\u202e💥"],
)
def test_formato_invalido_strings(entrada):
    r = consultar_estado_pedido(entrada)
    assert r["error"] == "formato_invalido"
    assert r["order_id"] is None
    assert "estado" not in r and "producto" not in r
    assert r["mensaje"]


@pytest.mark.parametrize(
    "entrada",
    [None, 1001, 10.5, True, False, [], ["ORD-1001"], {}, {"order_id": "ORD-1001"},
     b"ORD-1001", object(), ("ORD-1001",)],
)
def test_formato_invalido_tipos_raros(entrada):
    r = consultar_estado_pedido(entrada)  # type: ignore[arg-type]
    assert r["error"] == "formato_invalido"
    assert r["order_id"] is None


def test_string_enorme_no_falla_ni_se_refleja():
    r = consultar_estado_pedido("ORD-" + "1" * 1_000_000)
    assert r["error"] == "formato_invalido"
    assert len(json.dumps(r)) < 1000
    r = consultar_estado_pedido(" " * 1_000_000 + "ORD-1001")
    assert r["error"] == "formato_invalido"


@pytest.mark.parametrize("entrada", ["ORD-1001", "ORD-9999", "x", None, 5])
def test_resultado_serializable_json(entrada):
    json.dumps(consultar_estado_pedido(entrada))  # type: ignore[arg-type]


def test_firma_exacta():
    sig = inspect.signature(consultar_estado_pedido)
    assert list(sig.parameters) == ["order_id"]
    assert sig.parameters["order_id"].annotation in (str, "str")
    assert sig.return_annotation in (dict, "dict")


def test_repositorio_cumple_puerto_y_es_inmutable():
    repo = OrderRepositoryMock()
    assert isinstance(repo, OrderRepository)
    r = repo.consultar_estado_pedido("ORD-1001")
    r["estado"] = "alterado"
    assert repo.consultar_estado_pedido("ORD-1001")["estado"] == "En tránsito"
