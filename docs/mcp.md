# Servidor MCP de TiendaHogar

Expone por MCP (transporte stdio) la misma tool que usa el agente:
`consultar_estado_pedido(order_id: str) -> dict`. Es la misma función de
`src/tiendahogar_agent/pedidos.py`, sin lógica duplicada. Devuelve el pedido, o un dict con
`error` (`no_encontrado`, `formato_invalido`, `error_interno`). Datos: tabla mock ORD-1001 a ORD-1004.

## Ejecución

Con el entorno virtual activado y el paquete instalado (`python -m pip install -e ".[dev]"`):

```
python -m tiendahogar_agent.mcp_server
```

En stdio solo se escribe el protocolo en stdout; los logs salen por stderr.

## Probar con MCP Inspector

```
npx @modelcontextprotocol/inspector .venv\Scripts\python.exe -m tiendahogar_agent.mcp_server
```

En Git Bash, Mac y Linux usa `.venv/bin/python` (o `.venv/Scripts/python` en Git Bash). Si no instalaste el paquete con `-e`, define antes
`PYTHONPATH=src`. En la interfaz: Connect, pestaña Tools, List Tools y ejecuta
`consultar_estado_pedido` con `{"order_id": "ORD-1001"}`.

## Configurar en Claude Desktop

Edita `claude_desktop_config.json` (Windows: `%APPDATA%\Claude\`; Mac:
`~/Library/Application Support/Claude/`) con la ruta absoluta al python del venv:

```json
{
  "mcpServers": {
    "tiendahogar": {
      "command": "C:\\ruta\\al\\repo\\.venv\\Scripts\\python.exe",
      "args": ["-m", "tiendahogar_agent.mcp_server"],
      "env": { "PYTHONPATH": "C:\\ruta\\al\\repo\\src" }
    }
  }
}
```

En Windows las barras invertidas van escapadas, es decir, dobles (`\\`); también valen barras normales (`C:/ruta/al/repo/...`). En Mac/Linux usa
`/ruta/al/repo/.venv/bin/python` y `/ruta/al/repo/src`. `PYTHONPATH` no hace falta si
instalaste el paquete con `pip install -e .` en ese venv. Reinicia Claude Desktop tras editar.

## Pruebas

`python -m pytest tests/test_mcp.py -q` (sesión cliente en memoria, sin red).
