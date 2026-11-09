import json
import logging
import re
from collections.abc import Callable
from typing import Any, Literal, Protocol

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.product import NO_DATA, ProductInput
from app.settings import Settings

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
MAX_TOOL_CALLS_PER_RESPONSE = 4
SEO_TITLE_LENGTH = (1, 70)
SEO_DESCRIPTION_LENGTH = (800, 1200)
COMMERCIAL_BLOCK_MIN_LENGTH = 1
COUNT_CHARACTERS_TOOL = {
    "type": "function",
    "function": {
        "name": "count_characters",
        "description": (
            "Считает число символов в переданном тексте так же, как Python len()."
        ),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class ProviderError(RuntimeError):
    """Raised when the selected LLM provider cannot complete a request."""


class GenerationError(RuntimeError):
    """Raised when a provider response cannot satisfy the product-card contract."""


class GeneratedFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seo_title: str = Field(
        min_length=SEO_TITLE_LENGTH[0],
        max_length=SEO_TITLE_LENGTH[1],
    )
    seo_description: str = Field(
        min_length=SEO_DESCRIPTION_LENGTH[0],
        max_length=SEO_DESCRIPTION_LENGTH[1],
    )
    search_queries: list[str] = Field(min_length=3, max_length=5)
    commercial_block: str = Field(min_length=COMMERCIAL_BLOCK_MIN_LENGTH)

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
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты SEO-копирайтер интернет-магазина. Для каждого товара ты создаёшь SEO-заголовок и SEO-описание."
                    "# ЖЁСТКИЕ ОГРАНИЧЕНИЯ (нарушение = брак, результат будет отклонён автоматически) "
                    "Составь на русском языке только четыре поля черновика: "
                    "1. seo_title: СТРОГО не более 70 символов. Целевая длина: 55–65 символов. "
                    "2. seo_description: СТРОГО от 800 до 1200 символов. Целевая длина: 900–1100 символов."
                    "3. search_queries: СТРОГО от 3 до 5 поисковых запросов, которые логично использовать для оптимизации страницы товара. "
                    "4. commercial_block: Краткое описание преимуществ и целевой аудитории для оптимизации страницы товара. "
                    "Длина считается в символах, включая пробелы, знаки препинания и специальные символы. "
                    "Для точной проверки длины вызывай инструмент count_characters для seo_title и seo_description. "
                    "Если длина вне требуемого диапазона, исправь текст и снова проверь его инструментом. "
                    "Повторяй проверку после каждого исправления; не оценивай длину на глаз. "
                    "Для seo_title допустимо не более 70 символов; для seo_description — от 800 до 1200 символов включительно. "
                    "Верни только JSON-объект без дополнительных полей. "
                    "Используй исключительно факты из переданных данных; "
                    "не добавляй внешние сведения, не делай неподтверждённых выводов и не меняй значения характеристик. "
                    "Если данных недостаточно для правдивого seo_description "
                    "нужной длины или для написания commercial_block, то откажись от генерации. "
                    "commercial_block должен содержать ответ на вопрос: кому подходит прибор и какие задачи решает? "
                    "Не заполняй объём повторами или неподтверждёнными утверждениями. "
                    "Сначала создай черновик, проверь длину инструментом и исправляй поля до попадания "
                    "в диапазоны. После проверок верни только полный JSON-объект."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(product.model_dump(), ensure_ascii=False),
            },
        ]

        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            response = await self._create_completion(
                messages,
                tool_choice="required" if tool_round == 0 else "auto",
            )
            if not response.choices:
                raise GenerationError("LLM-провайдер не вернул результат.")

            message = response.choices[0].message
            if message.refusal:
                raise GenerationError(
                    "Недостаточно данных для достоверной генерации карточки."
                )
            if message.tool_calls:
                if tool_round == MAX_TOOL_ROUNDS:
                    raise GenerationError(
                        "LLM-провайдер не завершил проверку длины карточки."
                    )
                if len(message.tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
                    raise GenerationError(
                        "LLM-провайдер запросил слишком много вызовов инструмента."
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in message.tool_calls
                        ],
                    }
                )
                for tool_call in message.tool_calls:
                    if tool_call.function.name != "count_characters":
                        raise GenerationError(
                            "LLM-провайдер запросил неизвестный инструмент."
                        )
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as error:
                        raise GenerationError(
                            "LLM-провайдер передал некорректные аргументы инструмента."
                        ) from error
                    if (
                        not isinstance(arguments, dict)
                        or set(arguments) != {"text"}
                        or not isinstance(arguments.get("text"), str)
                    ):
                        raise GenerationError(
                            "LLM-провайдер передал некорректные аргументы инструмента."
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {"character_count": len(arguments["text"])}
                            ),
                        }
                    )
                continue

            if message.content is None:
                raise GenerationError("LLM-провайдер не вернул содержимое карточки.")
            return message.content

        raise GenerationError(
            "LLM-провайдер не завершил проверку длины карточки."
        )

    async def _create_completion(
        self,
        messages: list[dict[str, object]],
        tool_choice: Literal["auto", "required"],
    ) -> ChatCompletion:
        try:
            return await self._client.chat.completions.create(
                model=self._model,
                max_tokens=50_000,
                response_format={"type": "json_object"},
                tools=[COUNT_CHARACTERS_TOOL],
                tool_choice=tool_choice,
                messages=messages,
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


def _extract_single_json_object(raw_response: str) -> dict[str, Any]:
    if not raw_response or not raw_response.strip():
        raise json.JSONDecodeError("empty response", raw_response, 0)

    stripped = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced is not None:
        stripped = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict[str, Any], int]] = []

    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, consumed = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append((index, parsed, consumed))

    if not candidates:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)

    selected_start, selected_object, selected_consumed = candidates[0]
    for next_start, _, _ in candidates[1:]:
        if next_start < selected_start + selected_consumed:
            continue
        raise ValueError("Несколько JSON-объектов в ответе LLM-провайдера.")

    return selected_object


async def generate_card(
    product: ProductInput,
    provider: CompletionProvider,
) -> ProductCard:
    response = await provider.complete(product)

    try:
        generated_data = _extract_single_json_object(response)
    except (json.JSONDecodeError, ValueError) as error:
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
