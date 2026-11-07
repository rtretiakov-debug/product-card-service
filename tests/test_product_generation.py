import asyncio
import json
from html.parser import HTMLParser

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


class FormResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = None
        self._inside_result = False
        self._result_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre" and self.result is None:
            self._inside_result = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._inside_result:
            self.result = "".join(self._result_parts)
            self._inside_result = False

    def handle_data(self, data: str) -> None:
        if self._inside_result:
            self._result_parts.append(data)


async def post_product(
    app: FastAPI,
    payload: dict[str, object],
    params: dict[str, str] | None = None,
) -> Response:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/products/generate",
                json=payload,
                params=params or {},
            )


async def post_form(app: FastAPI, data: dict[str, str]) -> Response:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post("/", data=data)


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


def test_generate_product_card_supports_markdown_output(
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
            {"name": "Манометр", "manufacturer": "", "description": ""},
            params={"format": "markdown"},
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "# Манометр" in response.text
    assert "## Характеристики" in response.text


def test_product_form_mentions_provider_notice_and_renders_markdown_result(
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

    async def check() -> None:
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                page = await client.get("/")
                assert page.status_code == 200
                assert "LLM-провайдеру" in page.text
                form_data = {
                    "name": "Манометр DM 02",
                    "manufacturer": "",
                    "description": "Точное описание",
                    "specs": '{"диапазон измерений": "0–1,6 МПа"}',
                }
                form_response = await client.post(
                    "/",
                    data={**form_data, "format": "markdown"},
                )
                assert form_response.status_code == 200
                assert "# Манометр DM 02" in form_response.text
                assert "Введённые данные будут переданы" in form_response.text

    asyncio.run(check())


def test_product_form_returns_full_json_card_with_cyrillic(
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
        post_form(
            app,
            {
                "name": "Манометр DM 02",
                "manufacturer": "",
                "description": "Исходное описание прибора.",
                "specs": '{"диапазон измерений": "0–1,6 МПа", "точность": ""}',
                "format": "json",
            },
        )
    )

    assert response.status_code == 200
    assert '<div class="error">' not in response.text
    assert "Манометр DM 02" in response.text
    assert "диапазон измерений" in response.text
    assert "\\u" not in response.text

    parser = FormResultParser()
    parser.feed(response.text)
    assert parser.result is not None
    result = json.loads(parser.result)

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
    assert result["seo_title"] == "Манометр DM 02"
    assert result["seo_description"] == valid_generated_fields()["seo_description"]
    assert result["search_queries"] == valid_generated_fields()["search_queries"]
    assert result["commercial_block"] == valid_generated_fields()["commercial_block"]
    assert result["name"] == "Манометр DM 02"
    assert result["manufacturer"] == "«данных нет»"
    assert result["specs"] == {
        "диапазон измерений": "0–1,6 МПа",
        "точность": "«данных нет»",
    }
    assert result["description"] == "Исходное описание прибора."


def test_product_form_shows_validation_error_without_success_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(provider_factory=lambda _settings: ControlledProvider())

    response = asyncio.run(
        post_form(
            app,
            {
                "name": "Манометр",
                "manufacturer": "",
                "description": "",
                "specs": '["не объект"]',
                "format": "json",
            },
        )
    )

    assert response.status_code == 200
    assert "Поле specs должно быть JSON-объектом." in response.text
    assert '<div class="error">' in response.text
    assert '<div class="result">' not in response.text


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
def test_product_form_shows_generation_error_without_success_result(
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
        post_form(
            app,
            {
                "name": "Манометр",
                "manufacturer": "",
                "description": "",
                "format": "json",
            },
        )
    )

    assert response.status_code == 200
    assert message in response.text
    assert '<div class="error">' in response.text
    assert '<div class="result">' not in response.text


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
