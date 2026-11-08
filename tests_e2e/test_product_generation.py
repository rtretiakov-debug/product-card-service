import json

import pytest
from fastapi.testclient import TestClient

from app.main import app as live_app
from app.main import create_app
from app.product import ProductInput


class InvalidFieldProvider:
    async def complete(self, product: ProductInput) -> str:
        return json.dumps(
            {
                "seo_title": "Т" * 71,
                "seo_description": "О" * 800,
                "search_queries": [
                    "манометр",
                    "манометр прибор",
                    "манометр измерительный",
                ],
                "commercial_block": "Проверьте характеристики прибора.",
            },
            ensure_ascii=False,
        )

    async def close(self) -> None:
        pass


def test_invalid_provider_field_returns_502_and_logs_field(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    app = create_app(provider_factory=lambda _settings: InvalidFieldProvider())

    with TestClient(app) as client:
        response = client.post(
            "/products/generate",
            json={
                "name": "Манометр",
                "manufacturer": "Завод",
                "description": "Прибор для измерения давления.",
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Результат LLM-провайдера не соответствует требованиям к карточке."
        )
    }
    assert "Поле «seo_title»" in caplog.text
    assert "превышена максимальная длина (70 символов)" in caplog.text


def test_live_provider_generates_product_card_through_http_api() -> None:
    product = {
        "name": "Манометр показывающий, тестовый образец",
        "manufacturer": "Тестовая лаборатория",
        "specs": {
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "диаметр корпуса": "100 мм",
            "присоединительная резьба": "G1/2",
            "рабочая температура": "от −40 до +60 °C",
        },
        "description": (
            "Тестовый образец представляет собой показывающий манометр для "
            "измерения давления. В переданных характеристиках указан диапазон "
            "измерений от 0 до 1,6 МПа и класс точности 1,5. Диаметр корпуса "
            "составляет 100 мм, присоединительная резьба — G1/2. Указанный "
            "диапазон рабочей температуры окружающей среды составляет от "
            "−40 до +60 °C. При подготовке карточки название изделия, "
            "изготовитель, значения характеристик и это описание служат "
            "исходными сведениями о тестовом образце. Карточка предназначена "
            "для проверки передачи исходных характеристик и соблюдения "
            "ограничений формата ответа API. Технические параметры в запросе "
            "даны как тестовые данные и относятся только к этому образцу. "
            "Дополнительные сведения о материалах, сертификатах, комплекте "
            "поставки, сроке службы и совместимости с конкретным оборудованием "
            "в исходном описании не приведены."
        ),
    }

    with TestClient(live_app) as client:
        response = client.post(
            "/products/generate",
            json=product,
            timeout=180,
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    card = response.json()

    assert set(card) == {
        "seo_title",
        "seo_description",
        "search_queries",
        "commercial_block",
        "name",
        "manufacturer",
        "specs",
        "description",
    }
    assert isinstance(card["seo_title"], str) and card["seo_title"].strip()
    assert len(card["seo_title"]) <= 70
    assert isinstance(card["seo_description"], str)
    assert 800 <= len(card["seo_description"]) <= 1200
    assert 3 <= len(card["search_queries"]) <= 5
    assert all(
        isinstance(query, str) and query.strip()
        for query in card["search_queries"]
    )
    assert isinstance(card["commercial_block"], str)
    assert card["commercial_block"].strip()
    assert card["name"] == product["name"]
    assert card["manufacturer"] == product["manufacturer"]
    assert card["specs"] == product["specs"]
    assert card["description"] == product["description"]
