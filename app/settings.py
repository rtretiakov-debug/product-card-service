from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SETTING_NAMES = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_model": "OPENAI_MODEL",
}


class ConfigurationError(ValueError):
    """Raised when required provider settings are missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: AnyHttpUrl = Field(validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(validation_alias="OPENAI_MODEL")

    @field_validator("openai_api_key")
    @classmethod
    def api_key_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if not key:
            raise ValueError("must not be empty")
        return SecretStr(key)

    @field_validator("openai_model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("must not be empty")
        return model


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    try:
        return Settings(_env_file=env_file)
    except ValidationError as error:
        missing: set[str] = set()
        invalid: set[str] = set()

        for detail in error.errors():
            field_name = str(detail["loc"][0])
            setting_name = _SETTING_NAMES.get(field_name, field_name)
            if detail["type"] == "missing":
                missing.add(setting_name)
            else:
                invalid.add(setting_name)

        messages = []
        if missing:
            messages.append(f"missing required settings: {', '.join(sorted(missing))}")
        if invalid:
            messages.append(f"invalid settings: {', '.join(sorted(invalid))}")
        raise ConfigurationError(
            "Invalid application configuration (.env): " + "; ".join(messages)
        ) from None
