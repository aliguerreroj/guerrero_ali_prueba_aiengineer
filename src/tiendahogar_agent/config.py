"""Configuración de la aplicación: settings.yaml + variables de entorno (el entorno gana)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

RUTA_YAML_POR_DEFECTO = Path(__file__).resolve().parents[2] / "settings.yaml"


def ruta_yaml() -> Path:
    """Ruta del YAML; se puede sobrescribir con TIENDAHOGAR_SETTINGS_FILE."""
    return Path(os.environ.get("TIENDAHOGAR_SETTINGS_FILE") or RUTA_YAML_POR_DEFECTO)


class _FuenteYaml(PydanticBaseSettingsSource):
    """Lee los valores del YAML (si existe)."""

    def _datos(self) -> dict[str, Any]:
        ruta = ruta_yaml()
        if not ruta.is_file():
            return {}
        return yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}

    def get_field_value(self, field, field_name):  # pragma: no cover - no se usa
        return self._datos().get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._datos()


class Settings(BaseSettings):
    """Ajustes tipados. Los secretos son opcionales y sin valor por defecto."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["fake", "anthropic"] = "fake"
    llm_model: str = "claude-haiku-4-5-20251001"
    top_k: int = 3
    umbral_recuperacion: float = 0.3
    umbral_reembolso: float = 500
    timeout_llm_s: float = 30
    timeout_tool_s: float = 5
    anthropic_api_key: SecretStr | None = None
    # Chunking de documentos (T05)
    chunk_strategy: Literal["none", "fixed", "recursive", "auto"] = "auto"
    chunk_umbral_corto: int = 1000  # caracteres; auto no divide documentos <= umbral
    chunk_tamano: int = 500
    chunk_solape: int = 50
    # Embeddings (T06)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    @model_validator(mode="after")
    def _validar_embedding(self) -> Settings:
        if not self.embedding_model.strip():
            raise ValueError("embedding_model no puede estar vacío")
        return self

    @model_validator(mode="after")
    def _validar_chunking(self) -> Settings:
        if self.chunk_umbral_corto < 1 or self.chunk_tamano < 1:
            raise ValueError("chunk_umbral_corto y chunk_tamano deben ser >= 1")
        if not 0 <= self.chunk_solape < self.chunk_tamano:
            raise ValueError("chunk_solape debe estar entre 0 y chunk_tamano - 1")
        return self

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Orden de prioridad: init > entorno > .env > YAML > defaults.
        return (init_settings, env_settings, dotenv_settings, _FuenteYaml(settings_cls))


def cargar_settings() -> Settings:
    """Construye la configuración vigente."""
    return Settings()
