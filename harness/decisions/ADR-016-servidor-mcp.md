# ADR-016 — Servidor MCP por stdio

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
La tool `consultar_estado_pedido(order_id: str) -> dict` debe poder usarse también desde clientes MCP (por ejemplo Claude Desktop o MCP Inspector), no solo desde el bucle del orquestador. La firma es obligatoria y no se debe duplicar lógica.

## Decisión
Implementado en `src/tiendahogar_agent/mcp_server.py` y documentado en `docs/mcp.md`.
- **FastMCP por stdio**: `crear_servidor()` construye `FastMCP("tiendahogar-pedidos")` y `main()` lo ejecuta con `transport="stdio"`. Comando: `python -m tiendahogar_agent.mcp_server`.
- **La misma función, sin envoltorio**: se registra `pedidos.consultar_estado_pedido` directamente (`servidor.tool(name="consultar_estado_pedido")(consultar_estado_pedido)`), la misma que usa el orquestador. No hay lógica duplicada y la firma sigue siendo la exigida.
- **Logs a stderr**: `logging.basicConfig(stream=sys.stderr, ...)`. En stdio solo el protocolo puede escribir en stdout, y cualquier otra salida lo corrompería.
- **Solo la tool de pedidos**: el servidor no expone el chat ni el resto del agente. Así la superficie ofrecida a un cliente MCP es solo la consulta de la tabla mock.
- La prueba (`tests/test_mcp.py`) usa una sesión cliente en memoria, sin red.

## Alternativas consideradas
- Envolver la función con una lógica propia del servidor: duplicaría validación y formato de errores.
- Exponer también el chat como tool MCP: ampliaría la superficie y delegaría la redacción a otro cliente sin las capas de ADR-012.
- Transporte HTTP/SSE en lugar de stdio: stdio es el más simple para uso local y no abre puertos.

## Consecuencias
- Positivas: una sola implementación de la tool para el agente y para MCP; sin puertos de red; fácil de probar.
- Límites: sin autenticación (stdio asume un cliente local de confianza); expone únicamente la tabla mock de pedidos; un transporte remoto necesitaría autenticación y control de acceso, que no están implementados.
