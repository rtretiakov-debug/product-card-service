import asyncio
import json

import httpx
from openai import AsyncOpenAI

from app.generation import OpenAICompletionProvider
from app.product import ProductInput


def test_provider_sends_max_tokens_limit() -> None:
    async def run_test() -> dict[str, object]:
        request_body: dict[str, object] = {}

        def handle_request(request: httpx.Request) -> httpx.Response:
            request_body.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "seo_title": "Манометр",
                                        "seo_description": "Описание. " + "А" * 800,
                                        "search_queries": [
                                            "манометр",
                                            "манометр прибор",
                                            "манометр измерительный",
                                        ],
                                        "commercial_block": "Для измерения давления.",
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        )
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://provider.test/v1",
            http_client=http_client,
        )
        provider = OpenAICompletionProvider(client, "test-model")
        try:
            await provider.complete(
                ProductInput(
                    name="Манометр",
                    manufacturer="Завод",
                    description="Прибор для измерения давления.",
                )
            )
        finally:
            await provider.close()

        return request_body

    request_body = asyncio.run(run_test())

    assert request_body["model"] == "test-model"
    assert request_body["max_tokens"] == 16_000
