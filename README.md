# TiendaHogar — Agente de soporte

> Esqueleto: se completa en la tarea T25.

## Requisitos

## Instalación

## Variables de entorno

## Ejecución del agente

## Pruebas

## Estructura del repo

## Docker y Makefile

Resumen (detalle y equivalentes de cada target para PowerShell y bash en [`docs/docker.md`](docs/docker.md)):

```
docker build -t tiendahogar-agent .
docker run -it --rm --env-file .env tiendahogar-agent                      # CLI de chat
docker run --rm -p 8080:8080 --env-file .env tiendahogar-agent python -m uvicorn tiendahogar_agent.api:app --host 0.0.0.0 --port 8080
```

Targets de `make`: `install`, `test`, `lint`, `eval`, `chat`, `api`, `docker-build` y `docker-run`. La API key nunca se copia a la imagen: se pasa con `--env-file` o `-e`.
