"""Configuración de la aplicación: settings.yaml + variables de entorno (el entorno gana)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

MODELO_EMBEDDING_POR_DEFECTO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

    llm_provider: Literal["fake", "anthropic", "azure"] = "fake"
    llm_model: str = "claude-haiku-4-5-20251001"
    top_k: int = 3
    # Relevancia (T07): se aplica sobre los puntajes ORIGINALES, no sobre el RRF. PROVISIONALES (T19).
    umbral_bm25: float = 0.5
    umbral_semantico: float = 0.3
    umbral_reembolso: float = 500
    # Resiliencia (T11). T12 pasa timeout_llm_s y max_reintentos_llm al SDK (timeout y
    # max_retries del cliente); los reintentos con backoff los hace el SDK, no el dominio.
    timeout_llm_s: float = 30
    timeout_tool_s: float = 5
    max_reintentos_llm: int = 2
    llm_max_tokens: int = 1024
    anthropic_api_key: SecretStr | None = None
    # Azure OpenAI (opcionales; el «modelo» es el nombre del deployment)
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str | None = None
    # Chunking de documentos (T05)
    chunk_strategy: Literal["none", "fixed", "recursive", "auto"] = "auto"
    chunk_umbral_corto: int = 1000  # caracteres; auto no divide documentos <= umbral
    chunk_tamano: int = 500
    chunk_solape: int = 50
    # Embeddings (T06)
    embedding_model: str = MODELO_EMBEDDING_POR_DEFECTO

    @model_validator(mode="after")
    def _validar_embedding(self) -> Settings:
        if not self.embedding_model.strip():
            raise ValueError("embedding_model no puede estar vacío")
        return self

    @model_validator(mode="after")
    def _validar_recuperacion(self) -> Settings:
        if self.top_k < 1:
            raise ValueError("top_k debe ser >= 1")
        if self.umbral_bm25 < 0:
            raise ValueError("umbral_bm25 debe ser >= 0 (un BM25 <= 0 nunca es relevante)")
        if not -1.0 <= self.umbral_semantico <= 1.0:
            raise ValueError("umbral_semantico debe estar entre -1 y 1 (similitud coseno)")
        return self

    @model_validator(mode="after")
    def _validar_resiliencia(self) -> Settings:
        if not 0 < self.timeout_llm_s <= 120:
            raise ValueError("timeout_llm_s debe estar en (0, 120] segundos")
        if not 0 < self.timeout_tool_s <= 60:
            raise ValueError("timeout_tool_s debe estar en (0, 60] segundos")
        if self.llm_max_tokens < 1:
            raise ValueError("llm_max_tokens debe ser >= 1")
        if not 0 <= self.max_reintentos_llm <= 5:
            raise ValueError("max_reintentos_llm debe estar entre 0 y 5")
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
