"""Pruebas del servidor MCP: en memoria, sin red ni subprocesos."""

from __future__ import annotations

import asyncio
import json

from mcp.shared.memory import create_connected_server_and_client_session

from tiendahogar_agent.mcp_server import crear_servidor
from tiendahogar_agent.pedidos import consultar_estado_pedido


def _llamar(order_id):
    async def _run():
        servidor = crear_servidor()
        async with create_connected_server_and_client_session(servidor._mcp_server) as cliente:
            return await cliente.call_tool("consultar_estado_pedido", {"order_id": order_id})

    return asyncio.run(_run())


def _datos(resultado) -> dict:
    assert not resultado.isError
    return json.loads(resultado.content[0].text)


def test_lista_la_tool_con_descripcion_y_esquema():
    async def _run():
        async with create_connected_server_and_client_session(
            crear_servidor()._mcp_server
        ) as cliente:
            return await cliente.list_tools()

    tools = asyncio.run(_run()).tools
    assert [t.name for t in tools] == ["consultar_estado_pedido"]
    tool = tools[0]
    assert tool.description and "pedido" in tool.description.lower()
    assert tool.inputSchema["properties"]["order_id"]["type"] == "string"
    assert tool.inputSchema["required"] == ["order_id"]


def test_id_valido():
    assert _datos(_llamar("ORD-1001")) == consultar_estado_pedido("ORD-1001")
    assert _datos(_llamar("ORD-1001"))["producto"] == "Refrigeradora"


def test_id_inexistente():
    datos = _datos(_llamar("ORD-9999"))
    assert datos == consultar_estado_pedido("ORD-9999")
    assert datos["error"] == "no_encontrado"


def test_id_formato_invalido():
    datos = _datos(_llamar("pedido uno"))
    assert datos == consultar_estado_pedido("pedido uno")
    assert datos["error"] == "formato_invalido"


def test_la_tool_registrada_es_la_misma_funcion():
    tool = crear_servidor()._tool_manager.get_tool("consultar_estado_pedido")
    assert tool.fn is consultar_estado_pedido


def test_docs_mcp_json_es_valido():
    """El bloque JSON de Claude Desktop en docs/mcp.md debe poder copiarse tal cual."""
    import json
    import re
    from pathlib import Path

    texto = (Path(__file__).resolve().parents[1] / "docs" / "mcp.md").read_text(encoding="utf-8")
    bloques = re.findall(r"```json\n(.*?)```", texto, flags=re.DOTALL)
    assert bloques, "falta el bloque JSON de configuración"
    for bloque in bloques:
        config = json.loads(bloque)
        assert "tiendahogar" in config["mcpServers"]
