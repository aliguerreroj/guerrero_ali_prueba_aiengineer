"""Servidor MCP (stdio) que expone la tool `consultar_estado_pedido`.

Reutiliza exactamente la misma función que el orquestador (`pedidos.consultar_estado_pedido`):
no hay lógica duplicada. En stdio solo el protocolo puede escribir en stdout; los logs van a stderr.

Ejecución: `python -m tiendahogar_agent.mcp_server`
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from tiendahogar_agent.pedidos import consultar_estado_pedido

NOMBRE_SERVIDOR = "tiendahogar-pedidos"


def crear_servidor() -> FastMCP:
    """Construye el servidor MCP con la tool registrada (la función original, sin envoltorios)."""
    servidor = FastMCP(NOMBRE_SERVIDOR)
    servidor.tool(name="consultar_estado_pedido")(consultar_estado_pedido)
    return servidor


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    crear_servidor().run(transport="stdio")


if __name__ == "__main__":
    main()
