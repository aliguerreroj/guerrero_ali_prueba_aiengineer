# Atajos del proyecto. Equivalentes sin make (PowerShell y bash): docs/docker.md.
# Se usan rutas relativas entre comillas para que funcione con espacios en la ruta del repo.

# Python del .venv si existe (Linux/Mac o Windows); si no, el python del PATH.
# Se puede forzar: make test PYTHON=/ruta/a/python
PYTHON ?= $(shell if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"; elif [ -x ".venv/Scripts/python.exe" ]; then echo ".venv/Scripts/python.exe"; else echo python; fi)
IMAGE ?= tiendahogar-agent
PUERTO ?= 8080

.PHONY: install test lint eval chat api docker-build docker-run

install:
	"$(PYTHON)" -m pip install -e ".[dev]"

test:
	"$(PYTHON)" -m pytest tests/ -q

lint:
	"$(PYTHON)" -m ruff check .

# OJO: hace llamadas REALES a la API de Anthropic (cuesta dinero; requiere ANTHROPIC_API_KEY).
eval:
	"$(PYTHON)" -m tiendahogar_agent.evals

chat:
	"$(PYTHON)" -m tiendahogar_agent.cli

api:
	"$(PYTHON)" -m uvicorn tiendahogar_agent.api:app --port $(PUERTO)

docker-build:
	docker build -t $(IMAGE) .

# CLI interactiva en el contenedor; usa .env como --env-file si existe (la clave no entra a la imagen).
docker-run:
	docker run --rm -it $(if $(wildcard .env),--env-file .env,-e LLM_PROVIDER=fake) $(IMAGE)
