"""Pruebas de configuración (sin red ni API key)."""

import pytest
from pydantic import ValidationError

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
    assert s.top_k == 3
    assert s.llm_model == "claude-haiku-4-5-20251001"
    assert s.umbral_bm25 >= 0
    assert -1 <= s.umbral_semantico <= 1
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


def test_chunking_defaults_y_entorno(monkeypatch):
    s = Settings()
    assert s.chunk_strategy == "auto" and s.chunk_umbral_corto >= 400
    assert 0 <= s.chunk_solape < s.chunk_tamano
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed")
    monkeypatch.setenv("CHUNK_TAMANO", "200")
    s = Settings()
    assert s.chunk_strategy == "fixed" and s.chunk_tamano == 200


def test_chunking_valores_invalidos():
    with pytest.raises(ValidationError):
        Settings(chunk_strategy="magica")
    with pytest.raises(ValidationError):
        Settings(chunk_tamano=10, chunk_solape=10)


def test_recuperacion_valores_invalidos():
    with pytest.raises(ValidationError):
        Settings(umbral_bm25=-0.1)
    with pytest.raises(ValidationError):
        Settings(umbral_semantico=1.5)
    with pytest.raises(ValidationError):
        Settings(top_k=0)
    assert Settings(umbral_bm25=2, umbral_semantico=0.1).umbral_bm25 == 2
