import asyncio
import json
from html.parser import HTMLParser
from pathlib import Path

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


async def post_raw_product(app: FastAPI, content: str) -> Response:
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/products/generate",
                content=content,
                headers={"content-type": "application/json"},
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
    assert response.headers["content-type"].startswith("application/json")
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
    ("fields", "expected_field", "expected_reason"),
    [
        (
            {**valid_generated_fields(), "seo_title": "Т" * 71},
            "seo_title",
            "превышена максимальная длина (70 символов)",
        ),
        (
            {**valid_generated_fields(), "seo_description": "О" * 799},
            "seo_description",
            "длина меньше минимума (800 символов)",
        ),
        (
            {**valid_generated_fields(), "seo_description": "О" * 1201},
            "seo_description",
            "превышена максимальная длина (1200 символов)",
        ),
        (
            {**valid_generated_fields(), "search_queries": ["один", "два"]},
            "search_queries",
            "недостаточно элементов (минимум 3)",
        ),
        (
            {
                **valid_generated_fields(),
                "search_queries": ["запрос"] * 6,
            },
            "search_queries",
            "слишком много элементов (максимум 5)",
        ),
        (
            {**valid_generated_fields(), "unexpected": "field"},
            "unexpected",
            "неожиданное поле",
        ),
        (
            {**valid_generated_fields(), "search_queries": ["", "два", "три"]},
            "search_queries",
            "поисковые запросы не могут быть пустыми",
        ),
        (
            {
                **valid_generated_fields(),
                "search_queries": ["первый", 2, "третий"],
            },
            "search_queries.1",
            "ожидалась строка",
        ),
        (
            {
                key: value
                for key, value in valid_generated_fields().items()
                if key != "commercial_block"
            },
            "commercial_block",
            "обязательное поле отсутствует",
        ),
    ],
)
def test_generate_product_card_rejects_out_of_contract_content(
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, object],
    expected_field: str,
    expected_reason: str,
    caplog: pytest.LogCaptureFixture,
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
    assert expected_field in caplog.text
    assert expected_reason in caplog.text


def test_generate_product_card_logs_every_invalid_field(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    fields["seo_title"] = "Т" * 71
    fields["seo_description"] = "О" * 799
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

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Результат LLM-провайдера не соответствует требованиям к карточке."
    )
    assert "seo_title" in caplog.text
    assert "seo_description" in caplog.text


def test_generate_product_card_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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
    assert "JSON" in caplog.text


def test_generate_product_card_accepts_json_with_markdown_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    payload = (
        "Сделал карточку для вас.\n\n```json\n"
        f"{json.dumps(fields, ensure_ascii=False)}\n```\n\n"
        "Проверьте, пожалуйста."
    )
    app = create_app(provider_factory=lambda _settings: ControlledProvider(payload))

    response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "", "description": ""},
        )
    )

    assert response.status_code == 200
    assert response.json()["seo_title"] == fields["seo_title"]


def test_generate_product_card_accepts_raw_json_with_explanatory_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    payload = (
        "Ниже результат:\n"
        f"{json.dumps(fields, ensure_ascii=False)}\n"
        "Спасибо."
    )
    app = create_app(provider_factory=lambda _settings: ControlledProvider(payload))

    response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "", "description": ""},
        )
    )

    assert response.status_code == 200
    assert response.json()["seo_description"] == fields["seo_description"]


def test_generate_product_card_rejects_multiple_json_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    payload = (
        f"{json.dumps(fields, ensure_ascii=False)}\n"
        f"{json.dumps(fields, ensure_ascii=False)}"
    )
    app = create_app(provider_factory=lambda _settings: ControlledProvider(payload))

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
            {
                "name": "Манометр DM 02",
                "manufacturer": "Завод «Измеритель»",
                "specs": {"диапазон": "0–1,6 МПа", "точность": "1,5"},
                "description": "Паспорт прибора.\nТекст без изменений.",
            },
            params={"format": "markdown"},
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == (
        "# Манометр DM 02\n\n"
        "Производитель: Завод «Измеритель»\n\n"
        "## Характеристики\n"
        "- диапазон: 0–1,6 МПа\n"
        "- точность: 1,5\n\n"
        "## Описание\n"
        "Паспорт прибора.\nТекст без изменений.\n\n"
        "## SEO\n"
        "Заголовок: Манометр DM 02\n"
        f"Описание: {valid_generated_fields()['seo_description']}\n"
        "Поисковые запросы:\n"
        "- манометр DM 02\n"
        "- манометр Манотомь\n"
        "- манометр 0–1,6 МПа\n\n"
        "Commercial block:\n"
        "Проверьте характеристики прибора перед заказом."
    )


def test_generate_product_card_rejects_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(provider_factory=lambda _settings: ControlledProvider())

    response = asyncio.run(post_raw_product(app, "{"))

    assert response.status_code == 422
    assert any(
        error["type"] == "json_invalid"
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"manufacturer": "Завод", "description": "Описание"},
        {"name": " ", "manufacturer": "Завод", "description": "Описание"},
    ],
)
def test_generate_product_card_rejects_invalid_product_fields(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(provider_factory=lambda _settings: ControlledProvider())

    response = asyncio.run(post_product(app, payload))

    assert response.status_code == 422
    assert any("name" in error["loc"] for error in response.json()["detail"])


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


def test_product_form_renders_markdown_with_contract_boundaries_and_original_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    seo_title = "Т" * 70
    seo_description = "О" * 1200
    search_queries = [f"запрос {index}" for index in range(5)]
    fields["seo_title"] = seo_title
    fields["seo_description"] = seo_description
    fields["search_queries"] = search_queries
    provider = ControlledProvider(json.dumps(fields, ensure_ascii=False))
    app = create_app(provider_factory=lambda _settings: provider)
    description = "  Паспорт <прибора> & исходный текст\n"

    response = asyncio.run(
        post_form(
            app,
            {
                "name": "Манометр DM 02",
                "manufacturer": " ",
                "description": description,
                "specs": '{"точность": "", "диапазон": "0–1,6 МПа"}',
                "format": "markdown",
            },
        )
    )

    assert response.status_code == 200
    assert '<div class="error">' not in response.text
    parser = FormResultParser()
    parser.feed(response.text)
    assert parser.result is not None
    assert parser.result == (
        "# Манометр DM 02\n\n"
        "Производитель: «данных нет»\n\n"
        "## Характеристики\n"
        "- точность: «данных нет»\n"
        "- диапазон: 0–1,6 МПа\n\n"
        "## Описание\n"
        f"{description}\n\n"
        "## SEO\n"
        f"Заголовок: {seo_title}\n"
        f"Описание: {seo_description}\n"
        "Поисковые запросы:\n"
        + "\n".join(f"- {query}" for query in search_queries)
        + "\n\nCommercial block:\n"
        "Проверьте характеристики прибора перед заказом."
    )
    assert provider.received is not None
    assert provider.received.description == description
    assert provider.received.manufacturer == "«данных нет»"
    assert provider.received.specs == {
        "точность": "«данных нет»",
        "диапазон": "0–1,6 МПа",
    }


def test_product_form_rejects_generated_content_outside_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    fields = valid_generated_fields()
    fields["seo_description"] = "О" * 799
    app = create_app(
        provider_factory=lambda _settings: ControlledProvider(
            json.dumps(fields, ensure_ascii=False)
        )
    )

    response = asyncio.run(
        post_form(
            app,
            {
                "name": "Манометр",
                "manufacturer": "Завод",
                "description": "Исходное описание",
                "format": "json",
            },
        )
    )

    assert response.status_code == 200
    assert "Результат LLM-провайдера не соответствует требованиям к карточке." in (
        response.text
    )
    assert '<div class="error">' in response.text
    assert '<div class="result">' not in response.text


def test_product_requests_leave_no_local_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.chdir(tmp_path)
    app = create_app(
        provider_factory=lambda _settings: ControlledProvider(
            json.dumps(valid_generated_fields(), ensure_ascii=False)
        )
    )

    api_response = asyncio.run(
        post_product(
            app,
            {"name": "Манометр", "manufacturer": "Завод", "description": "Паспорт"},
        )
    )
    form_response = asyncio.run(
        post_form(
            app,
            {
                "name": "Манометр",
                "manufacturer": "Завод",
                "description": "Паспорт",
                "format": "json",
            },
        )
    )

    assert api_response.status_code == 200
    assert form_response.status_code == 200
    assert list(tmp_path.iterdir()) == []
