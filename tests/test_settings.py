import asyncio
from pathlib import Path
import traceback

from httpx import ASGITransport, AsyncClient, Response
import pytest

from app.main import app, run
from app.settings import ConfigurationError, load_settings

_SETTING_NAMES = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")


async def request_health() -> Response:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")


async def start_application() -> None:
    async with app.router.lifespan_context(app):
        pass


def clear_provider_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_missing_settings_are_reported_without_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_provider_settings(monkeypatch)

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)

    message = str(error.value)
    assert "OPENAI_API_KEY" in message
    assert "OPENAI_BASE_URL" in message
    assert "OPENAI_MODEL" in message


def test_invalid_endpoint_is_reported_without_echoing_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_provider_settings(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    invalid_endpoint = "https://user:private-token@invalid host/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", invalid_endpoint)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    with pytest.raises(ConfigurationError, match="OPENAI_BASE_URL") as error:
        load_settings(env_file=None)

    assert invalid_endpoint not in "".join(traceback.format_exception(error.value))
    assert "private-token" not in "".join(traceback.format_exception(error.value))


def test_blank_required_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_provider_settings(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", " ")

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)

    assert "OPENAI_API_KEY" in str(error.value)
    assert "OPENAI_MODEL" in str(error.value)


def test_valid_settings_mask_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_provider_settings(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    settings = load_settings(env_file=None)

    assert settings.openai_api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(settings)


def test_settings_are_loaded_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_provider_settings(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-secret\n"
        "OPENAI_BASE_URL=https://api.openai.com/v1\n"
        "OPENAI_MODEL=test-model\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == "test-secret"
    assert str(settings.openai_base_url) == "https://api.openai.com/v1"
    assert settings.openai_model == "test-model"


def test_health_endpoint_starts_with_valid_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_provider_settings(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_fails_for_missing_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_provider_settings(monkeypatch)

    with pytest.raises(ConfigurationError, match="Invalid application configuration"):
        asyncio.run(start_application())


def test_server_binds_only_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, str, int]] = []
    monkeypatch.setattr(
        "app.main.uvicorn.run",
        lambda application, host, port: calls.append((application, host, port)),
    )

    run()

    assert calls == [(app, "127.0.0.1", 8000)]
