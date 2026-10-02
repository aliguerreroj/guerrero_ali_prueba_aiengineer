"""Pruebas de los modelos de dominio."""

import json

import pytest
from pydantic import ValidationError

from tiendahogar_agent.models import AgentResponse, Chunk, Order


def _resp(**kw):
    base = {"respuesta": "Hola", "accion": "responder", "fuentes": ["garantia.md"], "trace_id": "t-1"}
    base.update(kw)
    return AgentResponse(**base)


@pytest.mark.parametrize("accion", ["responder", "escalar", "pedir_dato"])
def test_acciones_validas(accion):
    assert _resp(accion=accion).accion == accion


def test_accion_invalida():
    with pytest.raises(ValidationError):
        _resp(accion="ignorar")


def test_canal_opcional_por_defecto():
    assert _resp().canal is None
    assert _resp(accion="escalar", canal="soporte@tiendahogar.example").canal == (
        "soporte@tiendahogar.example"
    )


def test_tipos_erroneos_respuesta():
    with pytest.raises(ValidationError):
        _resp(fuentes="garantia.md")
    with pytest.raises(ValidationError):
        _resp(respuesta=123)


def test_chunk_metadatos_por_defecto():
    c = Chunk(texto="x", doc_id="d1")
    assert c.metadatos == {}
    assert Chunk(texto="x", doc_id="d1").metadatos is not Chunk(texto="y", doc_id="d2").metadatos


def test_chunk_tipos_erroneos():
    with pytest.raises(ValidationError):
        Chunk(texto="x", doc_id="d1", metadatos=["a"])


def test_order_entrega_opcional():
    o = Order(order_id="ORD-1002", producto="Licuadora", estado="Entregado")
    assert o.entrega_estimada is None
    o2 = Order(order_id="ORD-1001", producto="Refrigeradora", estado="En tránsito", entrega_estimada="2026-10-08")
    assert o2.entrega_estimada == "2026-10-08"


def test_order_tipo_erroneo():
    with pytest.raises(ValidationError):
        Order(order_id=["x"], producto="a", estado="b")


@pytest.mark.parametrize(
    "obj",
    [
        _resp(),
        Chunk(texto="t", doc_id="d", metadatos={"pagina": 1}),
        Order(order_id="ORD-1", producto="p", estado="e"),
    ],
)
def test_ida_y_vuelta(obj):
    datos = obj.model_dump(mode="json")
    assert type(obj).model_validate(datos) == obj
    assert type(obj).model_validate_json(json.dumps(datos)) == obj
