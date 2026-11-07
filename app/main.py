import html
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.generation import (
    CompletionProvider,
    GenerationError,
    ProductCard,
    ProviderError,
    ProviderFactory,
    create_openai_provider,
    generate_card,
)
from app.product import ProductInput
from app.settings import load_settings


def product_card_to_markdown(card: ProductCard) -> str:
    lines = [
        f"# {card.name}",
        "",
        f"Производитель: {card.manufacturer}",
        "",
        "## Характеристики",
    ]

    if isinstance(card.specs, str):
        lines.append(f"- {card.specs}")
    else:
        for key, value in card.specs.items():
            lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Описание",
        card.description,
        "",
        "## SEO",
        f"Заголовок: {card.seo_title}",
        f"Описание: {card.seo_description}",
        "Поисковые запросы:",
    ])
    lines.extend(f"- {query}" for query in card.search_queries)
    lines.extend([
        "",
        "Commercial block:",
        card.commercial_block,
    ])
    return "\n".join(lines)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_form_page(
    *,
    error: str | None = None,
    values: dict[str, str] | None = None,
    result: str | None = None,
    format: Literal["json", "markdown"] = "json",
) -> HTMLResponse:
    values = values or {}
    name = _escape(values.get("name", ""))
    manufacturer = _escape(values.get("manufacturer", ""))
    description = _escape(values.get("description", ""))
    specs = _escape(values.get("specs", ""))
    output_text = _escape(result or "")
    message = "" if error is None else f"<div class=\"error\">{_escape(error)}</div>"
    result_block = "" if result is None else (
        "<div class=\"result\"><h2>Результат</h2><pre>"
        f"{output_text}</pre></div>"
    )
    selected_json = " checked" if format == "json" else ""
    selected_markdown = " checked" if format == "markdown" else ""

    html_page = f"""<!DOCTYPE html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\">
  <title>Генерация карточки товара</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 900px; }}
    .field {{ margin-bottom: 1rem; }}
    label {{ display: block; margin-bottom: 0.35rem; font-weight: 600; }}
    input, textarea, select {{ width: 100%; box-sizing: border-box; padding: 0.55rem; font: inherit; }}
    textarea {{ min-height: 150px; resize: vertical; }}
    .note {{ background: #f5f5f5; border-left: 4px solid #666; padding: 0.8rem 1rem; margin-bottom: 1rem; }}
    .error {{ background: #fff2f2; border-left: 4px solid #b00020; color: #7a0000; padding: 0.8rem 1rem; margin: 1rem 0; }}
    .result {{ margin-top: 1.5rem; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8f8f8; padding: 1rem; overflow: auto; }}
    .format-row {{ display: flex; gap: 1rem; margin: 1rem 0; }}
    .format-row label {{ font-weight: 400; display: inline-flex; align-items: center; gap: 0.4rem; }}
    button {{ padding: 0.75rem 1.25rem; font: inherit; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>Генерация карточки товара</h1>
  <div class=\"note\">
    Введённые данные будут переданы выбранному LLM-провайдеру. Сервис не хранит данные и не обещает политику хранения данных провайдера.
  </div>
  {message}
  <form method=\"post\" action=\"/\">
    <div class=\"field\">
      <label for=\"name\">Название</label>
      <input id=\"name\" name=\"name\" value=\"{name}\" required>
    </div>
    <div class=\"field\">
      <label for=\"manufacturer\">Производитель</label>
      <input id=\"manufacturer\" name=\"manufacturer\" value=\"{manufacturer}\">
    </div>
    <div class=\"field\">
      <label for=\"description\">Описание</label>
      <textarea id=\"description\" name=\"description\">{description}</textarea>
    </div>
    <div class=\"field\">
      <label for=\"specs\">Характеристики (JSON-объект)</label>
      <textarea id=\"specs\" name=\"specs\" placeholder='{{\"диапазон измерений\": \"0–1,6 МПа\"}}'>{specs}</textarea>
    </div>
    <div class=\"format-row\">
      <label><input type=\"radio\" name=\"format\" value=\"json\"{selected_json}>JSON</label>
      <label><input type=\"radio\" name=\"format\" value=\"markdown\"{selected_markdown}>Markdown</label>
    </div>
    <button type=\"submit\">Сгенерировать карточку</button>
  </form>
  {result_block}
</body>
</html>
"""
    return HTMLResponse(content=html_page)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    provider_factory: ProviderFactory = application.state.provider_factory
    provider: CompletionProvider = provider_factory(settings)
    application.state.provider = provider
    try:
        yield
    finally:
        await provider.close()


def create_app(provider_factory: ProviderFactory = create_openai_provider) -> FastAPI:
    application = FastAPI(title="Product Card Service", lifespan=lifespan)
    application.state.provider_factory = provider_factory

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", response_class=HTMLResponse)
    async def product_form() -> HTMLResponse:
        return _render_form_page()

    @application.post("/", response_class=HTMLResponse)
    async def submit_product_form(
        request: Request,
        name: str = Form(...),
        manufacturer: str = Form(""),
        description: str = Form(""),
        specs: str = Form(""),
        format: Literal["json", "markdown"] = Form("json"),
    ) -> HTMLResponse:
        raw_values = {
            "name": name,
            "manufacturer": manufacturer,
            "description": description,
            "specs": specs,
        }
        try:
            payload: dict[str, Any] = {
                "name": name,
                "manufacturer": manufacturer,
                "description": description,
            }
            specs_text = specs.strip()
            if specs_text:
                parsed_specs = json.loads(specs_text)
                if not isinstance(parsed_specs, dict):
                    raise ValueError("Поле specs должно быть JSON-объектом.")
                payload["specs"] = parsed_specs
            product = ProductInput.model_validate(payload)
            card = await generate_card(product, request.app.state.provider)
            result = (
                json.dumps(card.model_dump(), indent=2, ensure_ascii=False)
                if format == "json"
                else product_card_to_markdown(card)
            )
            return _render_form_page(values=raw_values, result=result, format=format)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return _render_form_page(
                error=str(error),
                values=raw_values,
                format=format,
            )
        except ProviderError as error:
            return _render_form_page(
                error=str(error),
                values=raw_values,
                format=format,
            )
        except GenerationError as error:
            return _render_form_page(
                error=str(error),
                values=raw_values,
                format=format,
            )

    @application.post("/products/generate", response_model=ProductCard)
    async def generate_product_card(
        product: ProductInput,
        request: Request,
        format: Literal["json", "markdown"] = "json",
    ) -> ProductCard | PlainTextResponse:
        try:
            card = await generate_card(product, request.app.state.provider)
            if format == "markdown":
                return PlainTextResponse(
                    product_card_to_markdown(card),
                    media_type="text/markdown; charset=utf-8",
                )
            return card
        except ProviderError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except GenerationError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return application


app = create_app()


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
