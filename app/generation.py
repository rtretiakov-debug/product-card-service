import json
import logging
from collections.abc import Callable
from typing import Literal, Protocol

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.product import NO_DATA, ProductInput
from app.settings import Settings

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when the selected LLM provider cannot complete a request."""


class GenerationError(RuntimeError):
    """Raised when a provider response cannot satisfy the product-card contract."""


class GeneratedFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seo_title: str = Field(min_length=1, max_length=70)
    seo_description: str = Field(min_length=800, max_length=1200)
    search_queries: list[str] = Field(min_length=3, max_length=5)
    commercial_block: str = Field(min_length=1)

    @field_validator("seo_title", "seo_description", "commercial_block")
    @classmethod
    def generated_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("search_queries")
    @classmethod
    def search_queries_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not query.strip() for query in value):
            raise ValueError("search queries must not be blank")
        return value


class ProductCard(GeneratedFields):
    name: str
    manufacturer: str
    specs: dict[str, str] | Literal[NO_DATA]
    description: str


class CompletionProvider(Protocol):
    async def complete(self, product: ProductInput) -> str: ...

    async def close(self) -> None: ...


ProviderFactory = Callable[[Settings], CompletionProvider]


class OpenAICompletionProvider:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(self, product: ProductInput) -> str:
        response = await self._create_completion(product)
        if not response.choices:
            raise GenerationError("LLM-провайдер не вернул результат.")

        message = response.choices[0].message
        if message.refusal:
            raise GenerationError(
                "Недостаточно данных для достоверной генерации карточки."
            )
        if message.content is None:
            raise GenerationError("LLM-провайдер не вернул содержимое карточки.")
        return message.content

    async def _create_completion(self, product: ProductInput) -> ChatCompletion:
        try:
            return await self._client.chat.completions.create(
                model=self._model,
                max_tokens=50_000,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Составь на русском языке только четыре поля черновика: "
                            "seo_title, seo_description, search_queries и commercial_block. "
                            "Верни только JSON-объект без дополнительных полей. Используй "
                            "исключительно факты из переданных данных; не добавляй внешние "
                            "сведения, не делай неподтверждённых выводов и не меняй значения "
                            "характеристик. Если данных недостаточно для правдивого SEO-описания "
                            "нужной длины или коммерческого блока, откажись от генерации. "
                            "seo_title должен содержать максимум до 70 символов!; seo_description должен содержать минимум 800 и максимум 1200 символов включительно! "
                            "search_queries должен содержать минимум 3 и максимум 5 запросов только по входным данным! "
                            "Не заполняй объём повторами или неподтверждёнными утверждениями. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            product.model_dump(), ensure_ascii=False
                        ),
                    },
                ],
            )
        except APIError as error:
            raise ProviderError("Ошибка при обращении к LLM-провайдеру.") from error

    async def close(self) -> None:
        await self._client.close()


def create_openai_provider(settings: Settings) -> OpenAICompletionProvider:
    return OpenAICompletionProvider(
        AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url),
        ),
        settings.openai_model,
    )


def _validation_error_message(detail: dict[str, object]) -> str:
    error_type = detail["type"]
    context = detail.get("ctx", {})
    if not isinstance(context, dict):
        context = {}

    if error_type == "string_too_long":
        return f"превышена максимальная длина ({context.get('max_length')} символов)"
    if error_type == "string_too_short":
        return f"длина меньше минимума ({context.get('min_length')} символов)"
    if error_type == "too_short":
        return f"недостаточно элементов (минимум {context.get('min_length')})"
    if error_type == "too_long":
        return f"слишком много элементов (максимум {context.get('max_length')})"
    if error_type == "extra_forbidden":
        return "неожиданное поле"
    if error_type == "missing":
        return "обязательное поле отсутствует"
    if error_type == "string_type":
        return "ожидалась строка"
    if error_type == "list_type":
        return "ожидался список"
    if error_type in {"dict_type", "model_type"}:
        return "ожидался JSON-объект"
    if error_type == "value_error":
        cause = str(context.get("error", ""))
        if cause == "must not be blank":
            return "значение не может быть пустым"
        if cause == "search queries must not be blank":
            return "поисковые запросы не могут быть пустыми"
    return "значение не соответствует требованиям"


async def generate_card(
    product: ProductInput,
    provider: CompletionProvider,
) -> ProductCard:
    try:
        response = await provider.complete(product)
        generated_data = json.loads(response)
    except json.JSONDecodeError as error:
        logger.warning(
            "Результат LLM-провайдера не соответствует требованиям к карточке. "
            "Структура ответа: некорректный JSON."
        )
        raise GenerationError(
            "Результат LLM-провайдера не соответствует требованиям к карточке."
        ) from error

    try:
        generated = GeneratedFields.model_validate(generated_data)
    except ValidationError as error:
        for detail in error.errors(include_input=False):
            field = ".".join(str(part) for part in detail["loc"]) or "ответ"
            logger.warning(
                "Результат LLM-провайдера не соответствует требованиям к карточке. "
                "Поле «%s»: %s",
                field,
                _validation_error_message(detail),
            )
        raise GenerationError(
            "Результат LLM-провайдера не соответствует требованиям к карточке."
        ) from error

    return ProductCard(
        **generated.model_dump(),
        name=product.name,
        manufacturer=product.manufacturer,
        specs=product.specs,
        description=product.description,
    )
