from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request

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

    @application.post("/products/generate", response_model=ProductCard)
    async def generate_product_card(
        product: ProductInput,
        request: Request,
    ) -> ProductCard:
        try:
            return await generate_card(product, request.app.state.provider)
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
