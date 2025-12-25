import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import httpx
from src.utils.logging_config import get_logger
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.evaluators.base import BaseEvaluator


class BaseLLMJudge(BaseEvaluator, ABC):
    """Base class for LLM-based judge evaluators.

    Subclasses must implement `evaluate(generated, reference)` and return a dict
    containing at least `{"method": <name>, "score": <float>}`. Optionally
    include an `explanation` string.

    This base class provides `call_llm` which accepts OpenAI-format `messages`,
    an optional `model_params` dict and arbitrary request-level `**kwargs`.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds; used as multiplier for exponential backoff

    def __init__(self, llm_cfg: Any, api_key: str):
        self.model = llm_cfg.model
        self.base_url = llm_cfg.base_url
        self.provider = getattr(llm_cfg, "provider", None)
        self.temperature = getattr(llm_cfg, "temperature", 0.0)
        self.timeout = getattr(llm_cfg, "timeout", 30)
        # global stream flag controlled from runtime.yaml
        self.stream = getattr(llm_cfg, "stream", False)

        # optional default model/request params supplied in runtime config
        # e.g. {"max_tokens": 256, "top_p": 0.9}
        self.default_model_params = dict(getattr(llm_cfg, "model_params", {}) or {})

        self.api_key = api_key

        # Async HTTP client used by AsyncOpenAI
        self._http_client = httpx.AsyncClient(verify=False)
        self._openai = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, http_client=self._http_client)
        self._logger = get_logger(__name__)

    @abstractmethod
    async def evaluate(self, generated: str, reference: str, **kwargs) -> Dict[str, Any]:
        """Evaluate pair and return a result dict: {"method": str, "score": float, ...}.

        Implementations should return an explanation string when available.
        """
        raise NotImplementedError()

    @retry(retry=retry_if_exception_type(RateLimitError),
           wait=wait_exponential(multiplier=RETRY_DELAY, min=RETRY_DELAY, max=30),
           stop=stop_after_attempt(MAX_RETRIES),
           reraise=True)
    async def call_llm(self, messages: List[Dict[str, str]], model_params: Optional[Dict[str, Any]] = None,
                       response_schema: Optional[Any] = None, **kwargs) -> Any:
        """Call OpenAI Chat Completions.

        - `messages` must be in OpenAI chat format (list of {"role","content"}).
        - `model_params` can override default model/request params (e.g. `max_tokens`).
        - `**kwargs` are forwarded to the request and take precedence.

        Returns the model text output (string). Handles streaming and non-streaming
        responses, retries on RateLimitError, and will detect truncated results
        (finish_reason == "length") and retry up to MAX_RETRIES with exponential backoff.
        """

        # extract internal attempt counter (used for truncation retry flow)
        attempt = kwargs.pop("_attempt", 0)

        # friendly prompt text for logs / retry messages
        prompt_text = None
        for m in messages:
            if m.get("role") == "user":
                prompt_text = m.get("content")
                break
        prompt_text = prompt_text or str(messages)

        # build request kwargs by merging: built-in base fields <- defaults <- call-time model_params <- kwargs
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # apply configured defaults from runtime (if any)
        if self.default_model_params:
            call_kwargs.update(self.default_model_params)

        # ensure common defaults are present if not overridden
        call_kwargs.setdefault("temperature", self.temperature)
        call_kwargs.setdefault("timeout", self.timeout)
        call_kwargs.setdefault("stream", self.stream)

        # caller-supplied overrides
        if model_params:
            call_kwargs.update(model_params)
        if kwargs:
            call_kwargs.update(kwargs)

        # If a response_schema (Pydantic model class or JSON schema dict) is
        # provided and the provider is vllm, inject the vLLM structured output
        # `response_format` body so vLLM returns structured JSON directly.
        try:
            if response_schema is not None and (self.provider == "vllm" or kwargs.get("provider") == "vllm"):
                # accept either a Pydantic model class or a plain JSON schema dict
                if hasattr(response_schema, "schema"):
                    schema_dict = response_schema.schema()
                elif isinstance(response_schema, dict):
                    schema_dict = response_schema
                else:
                    schema_dict = None

                if schema_dict is not None:
                    call_kwargs["response_format"] = {"type": "json_schema", "json_schema": schema_dict}
        except Exception:
            # don't fail the call if response schema handling doesn't work
            self._logger.debug("Failed to prepare vLLM response_format for schema=%s", str(response_schema))

        try:
            resp = await self._openai.chat.completions.create(**call_kwargs)

            if call_kwargs.get("stream"):
                # streaming: collect chunks
                full = ""
                finish_reason = None
                async for chunk in resp:
                    try:
                        choice0 = chunk.choices[0]
                        # delta content may be attribute or dict
                        delta = getattr(choice0, "delta", None) or (choice0.get("delta") if isinstance(choice0, dict) else None)
                        part = getattr(delta, "content", None) if delta is not None else None
                        if part is None and isinstance(delta, dict):
                            part = delta.get("content")
                        if part:
                            full += part
                        finish_reason = getattr(choice0, "finish_reason", None) or (choice0.get("finish_reason") if isinstance(choice0, dict) else None)
                    except Exception:
                        # skip malformed chunk
                        continue

                content = await self._validate_and_retry(messages, model_params, full, finish_reason, prompt_text, attempt, **kwargs)
                return await self._maybe_parse_schema(content, response_schema)
            else:
                # non-streaming
                choice0 = resp.choices[0]
                message_obj = getattr(choice0, "message", None) or (choice0.get("message") if isinstance(choice0, dict) else None)
                content = getattr(message_obj, "content", None) or (message_obj.get("content") if isinstance(message_obj, dict) else "")
                finish_reason = getattr(choice0, "finish_reason", None) or (choice0.get("finish_reason") if isinstance(choice0, dict) else None)

                content = await self._validate_and_retry(messages, model_params, content or "", finish_reason, prompt_text, attempt, **kwargs)
                return await self._maybe_parse_schema(content, response_schema)

        except RateLimitError:
            # tenacity decorator will handle retries for RateLimitError
            raise
        except Exception as e:
            # bubble up other exceptions after logging
            self._logger.exception("LLM call failed")
            raise

    async def _validate_and_retry(self, messages: List[Dict[str, str]], model_params: Optional[Dict[str, Any]],
                                  content: str, finish_reason: Optional[str], prompt_text: str, attempt: int, **kwargs) -> str:
        """Validate finish_reason and retry if truncated.

        Retries by calling `call_llm` again and passing `_attempt` to avoid
        infinite recursion in case of repeated truncation. `prompt_text` is
        included in logs to help debugging when truncation or unusual finish
        reasons occur.
        """
        # shorten prompt_text for logging to avoid huge output
        short_prompt = (prompt_text[:200] + "...") if prompt_text and len(prompt_text) > 200 else (prompt_text or "<empty prompt>")

        if finish_reason == "length":
            # truncated output
            self._logger.warning("Response truncated (attempt %d) for prompt: %s. Retrying...", attempt + 1, short_prompt)
            if attempt < self.MAX_RETRIES - 1:
                delay = self.RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
                return await self.call_llm(messages, model_params, **{**kwargs, "_attempt": attempt + 1})
            else:
                self._logger.warning("Max truncation retries exceeded for prompt: %s; returning truncated content.", short_prompt)
                return content

        # treat 'stop' or None as success; other finish reasons logged and returned
        if finish_reason in (None, "stop"):
            return content

        self._logger.warning("Unusual finish reason: %s for prompt: %s; returning content.", finish_reason, short_prompt)
        return content

    async def _maybe_parse_schema(self, content: str, response_schema: Optional[Any]) -> Any:
        """If a Pydantic model class or JSON schema was provided, try to parse
        `content` into the model. Return the model instance on success or the
        raw content string on failure.
        """
        if response_schema is None:
            return content

        # If response_schema is a Pydantic model class, attempt to parse JSON
        try:
            import json
            parsed = json.loads(content)
        except Exception:
            self._logger.warning("Response not valid JSON for schema parsing")
            return content

        # If schema is a pydantic class, instantiate it
        try:
            if hasattr(response_schema, "__fields__") or hasattr(response_schema, "parse_obj"):
                inst = response_schema(**parsed)
                return inst
            # if schema is a dict, we can't validate fully here — return parsed dict
            if isinstance(response_schema, dict):
                return parsed
        except Exception as e:
            self._logger.warning("Schema instantiation failed: %s", e)
            return content

        return content
