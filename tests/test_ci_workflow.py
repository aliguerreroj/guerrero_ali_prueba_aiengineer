"""Pruebas del workflow de CI (T24): estructura del YAML y helper de topes. Sin red."""

from __future__ import annotations

from pathlib import Path

import yaml
from tiempos import FACTOR_TOPE, tope

RAIZ = Path(__file__).resolve().parent.parent
RUTA = RAIZ / ".github" / "workflows" / "ci.yml"


def _cargar() -> dict:
    return yaml.safe_load(RUTA.read_text(encoding="utf-8"))


def _disparadores(wf: dict) -> dict:
    # PyYAML interpreta la clave `on` como booleano True.
    return wf.get("on", wf.get(True))


def _pasos(wf: dict) -> list[dict]:
    return [p for job in wf["jobs"].values() for p in job["steps"]]


def _comandos(wf: dict) -> str:
    return "\n".join(p.get("run", "") for p in _pasos(wf))


def test_dispara_en_push_y_pull_request():
    d = _disparadores(_cargar())
    assert "push" in d
    assert "pull_request" in d


def test_matriz_de_so_y_versiones():
    job = next(iter(_cargar()["jobs"].values()))
    m = job["strategy"]["matrix"]
    assert set(m["os"]) == {"ubuntu-latest", "windows-latest"}
    assert [str(v) for v in m["python-version"]] == ["3.11", "3.13"]


def test_instala_ruff_y_pytest():
    cmds = _comandos(_cargar())
    assert 'pip install -e ".[dev]"' in cmds
    assert "ruff check ." in cmds
    assert "pytest" in cmds


def test_excluye_integration():
    cmds = _comandos(_cargar())
    linea = next(c for c in cmds.splitlines() if "pytest" in c)
    assert "not integration" in linea


def test_no_usa_secretos_ni_init():
    texto = RUTA.read_text(encoding="utf-8")
    assert "secrets." not in texto
    assert "init.py" not in _comandos(_cargar())


def test_cache_de_pip():
    setups = [p for p in _pasos(_cargar()) if str(p.get("uses", "")).startswith("actions/setup-python")]
    assert setups
    assert all(p["with"].get("cache") == "pip" for p in setups)


def test_marcador_integration_existe_en_pyproject():
    assert "integration:" in (RAIZ / "pyproject.toml").read_text(encoding="utf-8")


def test_tope_es_generoso_siempre():
    assert tope(2) == 2 * FACTOR_TOPE
    assert FACTOR_TOPE >= 5
