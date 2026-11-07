import asyncio
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from httpx import Response

from app.generation import GenerationError, ProviderError
from app.main import create_app
from app.product import ProductInput


def valid_generated_fields() -> dict[str, object]:
    return {
        "seo_title": "Манометр DM 02",
        "seo_description": "Описание товара. " + "А" * 800,
        "search_queries": [
            "манометр DM 02",
            "манометр Манотомь",
            "манометр 0–1,6 МПа",
        ],
        "commercial_block": "Проверьте характеристики прибора перед заказом.",
    }


class ControlledProvider:
    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.received: ProductInput | None = None

    async def complete(self, product: ProductInput) -> str:
        self.received = product
        if self.error is not None:
            raise self.error
        return self.content

    async def close(self) -> None:
        pass


async def post_product(
    app: FastAPI, payload: dict[str, object]
) -> Response:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post("/products/generate", json=payload)


def test_generate_product_card_returns_exact_contract_and_preserves_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    generated_fields = valid_generated_fields()
    provider = ControlledProvider(json.dumps(generated_fields, ensure_ascii=False))
    app = create_app(provider_factory=lambda _settings: provider)

    response = asyncio.run(
        post_product(
            app,
            {
                "name": "Манометр DM 02",
                "manufacturer": " ",
                "specs": {"диапазон измерений": "0–1,6 МПа", "точность": ""},
                "description": "  Исходное описание\nбез изменений ",
            },
        )
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "seo_title",
        "seo_description",
        "search_queries",
        "commercial_block",
        "name",
        "manufacturer",
        "specs",
        "description",
    }
    assert result["seo_title"] == generated_fields["seo_title"]
    assert result["seo_description"] == generated_fields["seo_description"]
    assert result["search_queries"] == generated_fields["search_queries"]
    assert result["commercial_block"] == generated_fields["commercial_block"]
    assert result["name"] == "Манометр DM 02"
    assert result["manufacturer"] == "«данных нет»"
    assert result["specs"] == {
        "диапазон измерений": "0–1,6 МПа",
        "точность": "«данных нет»",
    }
    assert result["description"] == "  Исходное описание\nбез изменений "
    assert provider.received.model_dump() == {
        "name": "Манометр DM 02",
        "manufacturer": "«данных нет»",
        "description": "  Исходное описание\nбез изменений ",
        "specs": {
            "диапазон измерений": "0–1,6 МПа",
            "точность": "«данных нет»",
        },
    }


def test_generate_product_card_marks_absent_specifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(
        provider_factory=lambda _settings: ControlledProvider(
            json.dumps(valid_generated_fields(), ensure_ascii=False)
        )
    )

    response = asyncio.run(
        post_product(
            app,
            {
                "name": "Манометр",
                "manufacturer": "Завод",
                "description": "Описание",
            },
        )
    )

    assert response.status_code == 200
    assert response.json()["specs"] == "«данных нет»"


@pytest.mark.parametrize(
    ("title_length", "description_length", "query_count"),
    [(1, 800, 3), (70, 1200, 5)],
)
def test_generate_product_card_accepts_inclusive_content_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    title_length: int,
    description_length: int,
    query_count: int,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    fields["seo_title"] = "Т" * title_length
    fields["seo_description"] = "О" * description_length
    fields["search_queries"] = [f"запрос {index}" for index in range(query_count)]
    app = create_app(
        provider_factory=lambda _settings: ControlledProvider(
            json.dumps(fields, ensure_ascii=False)
        )
    )

    response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "", "description": ""},
        )
    )

    assert response.status_code == 200
    result = response.json()
    assert len(result["seo_title"]) == title_length
    assert len(result["seo_description"]) == description_length
    assert len(result["search_queries"]) == query_count


@pytest.mark.parametrize(
    "fields",
    [
        {**valid_generated_fields(), "seo_title": "Т" * 71},
        {**valid_generated_fields(), "seo_description": "О" * 799},
        {**valid_generated_fields(), "seo_description": "О" * 1201},
        {**valid_generated_fields(), "search_queries": ["один", "два"]},
        {
            **valid_generated_fields(),
            "search_queries": ["запрос"] * 6,
        },
        {**valid_generated_fields(), "unexpected": "field"},
        {**valid_generated_fields(), "search_queries": ["", "два", "три"]},
    ],
)
def test_generate_product_card_rejects_out_of_contract_content(
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, object],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    provider = ControlledProvider(json.dumps(fields, ensure_ascii=False))
    app = create_app(provider_factory=lambda _settings: provider)

    response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "", "description": ""},
        )
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Результат LLM-провайдера не соответствует требованиям к карточке."
    )


def test_generate_product_card_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(provider_factory=lambda _settings: ControlledProvider("{"))

    response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "", "description": ""},
        )
    )

    assert response.status_code == 502
    assert "не соответствует требованиям" in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            ProviderError("Ошибка при обращении к LLM-провайдеру."),
            "Ошибка при обращении к LLM-провайдеру.",
        ),
        (
            GenerationError(
                "Недостаточно данных для достоверной генерации карточки."
            ),
            "Недостаточно данных для достоверной генерации карточки.",
        ),
    ],
)
def test_generate_product_card_surfaces_provider_and_insufficient_data_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(
        provider_factory=lambda _settings: ControlledProvider(error=error)
    )

    response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "", "description": ""},
        )
    )

    assert response.status_code == 502
    assert response.json()["detail"] == message
