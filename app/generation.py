import json
from collections.abc import Callable
from typing import Literal, Protocol

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.product import NO_DATA, ProductInput
from app.settings import Settings


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


async def generate_card(
    product: ProductInput,
    provider: CompletionProvider,
) -> ProductCard:
    try:
        generated = GeneratedFields.model_validate(
            json.loads(await provider.complete(product))
        )
    except (json.JSONDecodeError, ValidationError) as error:
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
