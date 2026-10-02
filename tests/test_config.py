"""Pruebas de configuración (sin red ni API key)."""

import pytest

from tiendahogar_agent.config import Settings, ruta_yaml


@pytest.fixture(autouse=True)
def _entorno_limpio(monkeypatch, tmp_path):
    for v in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "TOP_K", "UMBRAL_REEMBOLSO"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)  # evita leer un .env real


def test_defaults_del_yaml_del_repo():
    s = Settings()
    assert s.llm_provider == "fake"
    assert s.umbral_reembolso == 500
    assert s.top_k > 0 and s.llm_model
    assert 0 <= s.umbral_recuperacion <= 1
    assert s.timeout_llm_s > 0 and s.timeout_tool_s > 0


def test_provider_fake_valido_y_predeterminado():
    assert Settings().llm_provider == "fake"
    assert Settings(llm_provider="fake").llm_provider == "fake"


def test_entorno_gana_al_yaml(monkeypatch, tmp_path):
    y = tmp_path / "otro.yaml"
    y.write_text("top_k: 9\numbral_reembolso: 100\n", encoding="utf-8")
    monkeypatch.setenv("TIENDAHOGAR_SETTINGS_FILE", str(y))
    assert ruta_yaml() == y
    s = Settings()
    assert s.top_k == 9 and s.umbral_reembolso == 100
    monkeypatch.setenv("TOP_K", "2")
    assert Settings().top_k == 2


def test_secretos_sin_default(monkeypatch):
    assert Settings().anthropic_api_key is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "valor-de-prueba")
    s = Settings()
    assert s.anthropic_api_key.get_secret_value() == "valor-de-prueba"
    assert "valor-de-prueba" not in repr(s)


def test_yaml_sin_claves():
    from tiendahogar_agent.config import RUTA_YAML_POR_DEFECTO

    assert "api_key" not in RUTA_YAML_POR_DEFECTO.read_text(encoding="utf-8").lower()
