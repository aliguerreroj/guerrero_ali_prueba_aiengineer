"""Formato vs. cifra en la verificación de salida (hallazgo 2, vivo-06): sin red ni API key."""

from __future__ import annotations

import pytest

from tiendahogar_agent.guardrail_output import R_CIFRA, verificar_salida


def _verificar(texto, resultado_tool=None, mensaje="¿Dónde está mi pedido?"):
    return verificar_salida(texto, "pedir_dato", [], [], resultado_tool, mensaje)


@pytest.mark.parametrize("formato", ["ORD-####", "ORD-XXXX", "ord-xxxx", "ORD - ####"])
def test_formato_sin_digitos_no_es_cifra_ni_id(formato):
    r = _verificar(f"Claro, ¿me compartes tu número de pedido? Tiene la forma {formato}.")
    assert r.ok, r.detalles
    assert r.reglas_fallidas == ()


@pytest.mark.parametrize("inventado", ["ORD-1234", "ORD-0000", "ord - 9999"])
def test_ejemplo_concreto_con_digitos_sigue_bloqueado(inventado):
    r = _verificar(f"¿Me das tu número de pedido? Por ejemplo {inventado}.")
    assert not r.ok and R_CIFRA in r.reglas_fallidas
    assert any("id de pedido" in d for d in r.detalles)
    assert r.accion == "escalar"


def test_id_real_del_contexto_sigue_pasando():
    r = _verificar(
        "Tu pedido ORD-1001 va en camino.",
        resultado_tool={"order_id": "ORD-1001", "estado": "enviado"},
        mensaje="estado de ORD-1001",
    )
    assert r.ok, r.detalles
