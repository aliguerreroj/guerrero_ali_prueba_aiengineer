"""Configuración de la aplicación: settings.yaml + variables de entorno (el entorno gana)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import SecretStr
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
    llm_model: str = "claude-sonnet-4-5"
    top_k: int = 4
    umbral_recuperacion: float = 0.3
    umbral_reembolso: float = 500
    timeout_llm_s: float = 30
    timeout_tool_s: float = 5
    anthropic_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Orden de prioridad: init > entorno > .env > YAML > defaults.
        return (init_settings, env_settings, dotenv_settings, _FuenteYaml(settings_cls))


def cargar_settings() -> Settings:
    """Construye la configuración vigente."""
    return Settings()
