"""DeepSeek-V4.1 serving glue for the ``dsv41`` chat encoding spec: prompt
encoding through the in-tree ``encoding_dsv41`` encoder. Output stays with the
deepseek-v41 / deepseekv41 text parsers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.srt.entrypoints.openai.chat_encoding_adapters import OPENAI_CHAT_WIRE
from sglang.srt.environ import envs
from sglang.srt.multimodal.media_io import ImageData

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

_OPENAI_FUNCTION_FIELD_ORDER = ("name", "description", "parameters")


def parse_dsv41_reasoning_effort(value: Any) -> Union[str, int, None]:
    """Map an API ``reasoning_effort`` onto what the V4.1 encoder accepts.

    Tiers pass through; an OpenAI float in [0, 0.99] becomes a 1-100 budget;
    an int budget only reaches here through ``chat_template_kwargs``; None
    means unsupported, and the caller applies its default.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 100 else None
    if isinstance(value, float):
        return min(100, max(1, round(value * 100))) if 0.0 <= value <= 0.99 else None
    if value in encoding_dsv41.REASONING_EFFORT_MAPPINGS:
        return value
    return None


def dsv41_tool_payload(tool: Any) -> dict[str, Any]:
    """The tool dict the V4.1 encoder renders verbatim into the prompt.

    Only fields the client sent, in the OpenAI field order; pydantic would
    otherwise add defaults (strict=false) and reorder keys by declaration.
    """
    payload = tool.model_dump(exclude_unset=True, exclude_none=True)
    function = dict(payload.get("function") or {})
    ordered = {
        k: function.pop(k) for k in _OPENAI_FUNCTION_FIELD_ORDER if k in function
    }
    ordered.update(function)
    payload["function"] = ordered
    return payload


def default_dsv41_reasoning_effort_from_env(raw: Optional[str]) -> Union[str, int]:
    """Parse ``SGLANG_DSV41_REASONING_EFFORT``; raises so a bad value fails at boot."""
    if raw is None or not raw.strip():
        return encoding_dsv41.DEFAULT_REASONING_EFFORT
    value: Any = int(raw) if raw.strip().isdigit() else raw.strip()
    effort = parse_dsv41_reasoning_effort(value)
    if effort is None:
        raise ValueError(
            f"Invalid SGLANG_DSV41_REASONING_EFFORT={raw!r}; expected one of "
            f"{list(encoding_dsv41.REASONING_EFFORT_MAPPINGS)} or an integer in [1, 100]"
        )
    return effort


class DeepSeekV41ChatEncodingAdapter:
    # The encoder sorts tool results by call order, so images come from its render.
    collects_openai_media = False
    history_tool_call_arguments = "strict"
    renders_response_format = False
    uncounted_prompt_tokens = 0
    wire = OPENAI_CHAT_WIRE

    def __init__(
        self,
        *,
        tokenizer: Any,
        image_token_id: Optional[int],
        encode_prompt_text: Callable[[str], list[int]],
    ):
        self._tokenizer = tokenizer
        self._image_token_id = image_token_id
        self._encode_prompt_text = encode_prompt_text
        # Parsed once: the env var is frozen for the server's lifetime, and a
        # bad value fails startup, not every request.
        self._default_effort = default_dsv41_reasoning_effort_from_env(
            envs.SGLANG_DSV41_REASONING_EFFORT.get()
        )
        self._unsupported_efforts_warned: set[str] = set()

    def validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        return None

    @property
    def output(self) -> None:
        return None

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        for msg in messages:
            if msg.get("content") is None:
                msg["content"] = ""
        messages, assistant_prefix = _split_assistant_prefix(
            messages, request.continue_final_message
        )

        # The encoder renders a system token even for empty content, so an
        # empty system message is only inserted to host the request tools.
        request_tools = request.tools if tools else None
        if request_tools:
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["tools"] = [dsv41_tool_payload(tool) for tool in request_tools]
        if request.task is not None:
            encoding_dsv41.attach_task_to_last_user_message(messages, request.task)

        prompt, media = encoding_dsv41.encode_messages(
            messages,
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=self._resolve_effort(request.reasoning_effort),
            return_multi_modal_data=True,
        )
        if media["images"]:
            if self._image_token_id is None:
                raise ValueError("image input is not supported for this model")
            image_data.extend(ImageData(url=image["url"]) for image in media["images"])
            prompt = prompt.replace(
                encoding_dsv41.IMAGE_PLACEHOLDER,
                self._tokenizer.convert_ids_to_tokens(self._image_token_id),
            )
        prompt_ids = self._encode_prompt_text(prompt)
        if assistant_prefix:
            prompt_ids = prompt_ids + self._encode_assistant_prefix(assistant_prefix)
        return prompt_ids

    def _resolve_effort(self, value: Any) -> Union[str, int]:
        """Request effort for the encoder; unsupported values warn once and fall back."""
        effort = parse_dsv41_reasoning_effort(value)
        if effort is not None:
            return effort
        if (
            value is not None
            and value != "none"
            and repr(value) not in self._unsupported_efforts_warned
        ):
            self._unsupported_efforts_warned.add(repr(value))
            logger.warning(
                "DeepSeek-V4.1 does not support reasoning_effort=%r; using the "
                "default %r (low/high/xhigh/max, a float in [0, 0.99], or an "
                "integer budget in [1, 100] via chat_template_kwargs are accepted).",
                value,
                self._default_effort,
            )
        return self._default_effort

    def _encode_assistant_prefix(self, assistant_prefix: str) -> list[int]:
        encoded = self._tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self._tokenizer.bos_token_id:
            encoded = encoded[1:]
        return encoded


def _split_assistant_prefix(
    messages: list[dict[str, Any]], continue_final_message: bool
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """A trailing assistant turn with text content is the prefix to continue
    under ``continue_final_message``, else it is re-sent as a user turn."""
    if not messages or messages[-1].get("role") != "assistant":
        return messages, None
    content = messages[-1].get("content")
    if not isinstance(content, str):
        return messages, None
    if continue_final_message:
        return messages[:-1], content
    messages[-1] = {"role": "user", "content": content}
    return messages, None
