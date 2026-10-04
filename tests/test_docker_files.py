"""Pruebas rápidas (sin Docker) de Dockerfile, .dockerignore y Makefile (T23)."""

from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DOCKERFILE = (RAIZ / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (RAIZ / ".dockerignore").read_text(encoding="utf-8")
MAKEFILE = (RAIZ / "Makefile").read_text(encoding="utf-8")
DOC = (RAIZ / "docs" / "docker.md").read_text(encoding="utf-8")


def _instrucciones(texto: str) -> list[str]:
    return [ln.strip() for ln in texto.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_imagen_base_python_313_slim():
    froms = [ln for ln in _instrucciones(DOCKERFILE) if re.match(r"FROM\s+\S+( AS \w+)?$", ln)]
    assert froms and all(ln.split()[1] == "python:3.13-slim" for ln in froms)


def test_etapa_final_corre_como_usuario_no_root():
    lineas = _instrucciones(DOCKERFILE)
    ultimo_from = max(i for i, ln in enumerate(lineas) if ln.upper().startswith("FROM "))
    usuarios = [ln.split()[1] for ln in lineas[ultimo_from:] if ln.upper().startswith("USER ")]
    assert usuarios and usuarios[-1] not in {"root", "0"}


def test_cmd_por_defecto_es_la_cli():
    cmd = [ln for ln in _instrucciones(DOCKERFILE) if ln.startswith("CMD ")][-1]
    assert "tiendahogar_agent.cli" in cmd


def test_dockerfile_no_copia_env_ni_contexto_privado():
    for ln in _instrucciones(DOCKERFILE):
        if ln.upper().startswith(("COPY ", "ADD ")):
            assert not re.search(r"(^|[\s/])\.env(\s|$)", ln), ln
            assert "harness/context" not in ln, ln
            assert "harness " not in ln.replace("harness/docs_checksums.json", ""), ln


def test_modelo_de_embeddings_se_descarga_en_el_build_con_cache_explicita():
    assert "FASTEMBED_CACHE_PATH=/opt/fastembed_cache" in DOCKERFILE
    assert "TextEmbedding" in DOCKERFILE
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE


def test_dockerignore_excluye_lo_requerido():
    lineas = {ln.strip() for ln in DOCKERIGNORE.splitlines()}
    for patron in (
        ".env", ".venv/", "harness/context/", "logs/", "__pycache__/", ".pytest_cache/",
        ".ruff_cache/", ".git/", "*.egg-info/",
    ):
        assert patron in lineas, patron


def test_makefile_tiene_los_ocho_targets():
    for target in ("install", "test", "lint", "eval", "chat", "api", "docker-build", "docker-run"):
        assert re.search(rf"^{re.escape(target)}:", MAKEFILE, re.MULTILINE), target


def test_makefile_rutas_con_espacios_y_python_del_venv():
    assert re.search(r"^PYTHON \?=", MAKEFILE, re.MULTILINE)
    assert ".venv" in MAKEFILE
    assert '"$(PYTHON)"' in MAKEFILE
    assert "'$(PYTHON)'" not in MAKEFILE


def test_api_usa_puerto_8080_y_host_externo():
    assert "EXPOSE 8080" in DOCKERFILE
    assert "--port $(PUERTO)" in MAKEFILE and "PUERTO ?= 8080" in MAKEFILE
    assert "-p 8080:8080" in DOC and "--host 0.0.0.0 --port 8080" in DOC
    assert "8000" not in DOC


def test_sin_claves_ni_secretos_en_los_archivos():
    patron = re.compile(r"sk-ant-|sk-[A-Za-z0-9]{20,}|ANTHROPIC_API_KEY=[A-Za-z0-9]|AZURE_OPENAI_API_KEY=[A-Za-z0-9]")
    for nombre, texto in (("Dockerfile", DOCKERFILE), (".dockerignore", DOCKERIGNORE),
                          ("Makefile", MAKEFILE)):
        assert not patron.search(texto), nombre
    assert not re.search(r"^\s*(ENV|ARG)\s+\S*(KEY|SECRET|TOKEN)", DOCKERFILE, re.MULTILINE)
