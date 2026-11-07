from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from openai import AsyncOpenAI

from app.settings import load_settings


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    application.state.provider = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=str(settings.openai_base_url),
    )
    application.state.model = settings.openai_model
    try:
        yield
    finally:
        await application.state.provider.close()


def create_app() -> FastAPI:
    application = FastAPI(title="Product Card Service", lifespan=lifespan)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
