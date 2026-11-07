import pytest
from pydantic import ValidationError

from app.product import ProductInput


def test_product_input_requires_strings_and_preserves_empty_description() -> None:
    product = ProductInput.model_validate(
        {
            "name": "Манометр DM 02",
            "manufacturer": "",
            "description": "",
        }
    )

    assert product.model_dump() == {
        "name": "Манометр DM 02",
        "manufacturer": "«данных нет»",
        "description": "",
        "specs": "«данных нет»",
    }


def test_product_input_preserves_nonempty_name_and_description_verbatim() -> None:
    product = ProductInput.model_validate(
        {
            "name": "  Манометр DM 02  ",
            "manufacturer": "Завод",
            "description": " Исходный текст\n",
        }
    )

    assert product.name == "  Манометр DM 02  "
    assert product.description == " Исходный текст\n"


def test_product_input_normalizes_blank_manufacturer() -> None:
    product = ProductInput.model_validate(
        {
            "name": "Манометр DM 02",
            "manufacturer": "   ",
            "description": "",
        }
    )

    assert product.manufacturer == "«данных нет»"


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        ({}, "«данных нет»"),
        ("«данных нет»", "«данных нет»"),
        (
            {"произвольная характеристика": "", "точность": "1,5"},
            {"произвольная характеристика": "«данных нет»", "точность": "1,5"},
        ),
        (
            {"неизвестная характеристика": "   "},
            {"неизвестная характеристика": "«данных нет»"},
        ),
    ],
)
def test_product_input_normalizes_unknown_and_empty_specs(
    specs: object, expected: object
) -> None:
    product = ProductInput.model_validate(
        {
            "name": "Манометр DM 02",
            "manufacturer": "Завод",
            "description": "",
            "specs": specs,
        }
    )

    assert product.specs == expected


@pytest.mark.parametrize(
    "specs",
    [None, [], "неизвестное значение", {"точность": 1}],
)
def test_product_input_rejects_unsupported_specs(specs: object) -> None:
    with pytest.raises(ValidationError) as error:
        ProductInput.model_validate(
            {
                "name": "Манометр DM 02",
                "manufacturer": "Завод",
                "description": "",
                "specs": specs,
            }
        )

    assert "specs" in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"manufacturer": "Завод", "description": "Описание"},
        {"name": 123, "manufacturer": "Завод", "description": "Описание"},
        {"name": "", "manufacturer": "Завод", "description": "Описание"},
        {"name": " ", "manufacturer": "Завод", "description": "Описание"},
        {"name": "Манометр", "description": "Описание"},
        {"name": "Манометр", "manufacturer": "Завод"},
        {"name": "Манометр", "manufacturer": None, "description": "Описание"},
        {"name": "Манометр", "manufacturer": "Завод", "description": None},
        {"name": "Манометр", "manufacturer": 123, "description": "Описание"},
        {"name": "Манометр", "manufacturer": "Завод", "description": 123},
    ],
)
def test_product_input_rejects_missing_or_invalid_required_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ProductInput.model_validate(payload)
