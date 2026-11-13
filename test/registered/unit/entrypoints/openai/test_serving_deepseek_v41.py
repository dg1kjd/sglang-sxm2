"""The V100 port's DeepSeek-V4.1 chat encoding: the in-tree encoder behind the
dsv41 adapter, with output left to the text parsers."""

import copy
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 129264
IMAGE_TOKEN = "<｜image｜>"


class _CharTokenizer:
    """One id per character, so prompt ids decode back to the exact prompt."""

    bos_token_id = None
    chat_template = None

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def convert_ids_to_tokens(self, token_id):
        return {IMAGE_TOKEN_ID: IMAGE_TOKEN}[token_id]


def _decode(ids):
    return _CharTokenizer().decode(ids)


def _adapter(image_token_id=IMAGE_TOKEN_ID):
    from sglang.srt.entrypoints.openai.serving_deepseek_v41 import (
        DeepSeekV41ChatEncodingAdapter,
    )

    tokenizer = _CharTokenizer()
    return DeepSeekV41ChatEncodingAdapter(
        tokenizer=tokenizer,
        image_token_id=image_token_id,
        encode_prompt_text=tokenizer.encode,
    )


def _request(messages, **fields):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    return ChatCompletionRequest(messages=messages, **fields)


def _encode(adapter, request, *, thinking=False, image_data=None):
    """Encode as the handler does: the message dumps, the tool dumps."""
    tools = [tool.model_dump() for tool in request.tools] if request.tools else None
    return adapter.encode_messages(
        messages=[message.model_dump() for message in request.messages],
        request=request,
        tools=tools,
        thinking=thinking,
        image_data=[] if image_data is None else image_data,
    )


class TestDsv41Encoder(CustomTestCase):
    def test_dsv41_rejects_image_placeholder_in_text(self):
        for role in ("user", "tool"):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    encoding_dsv41.encode_messages(
                        [
                            {
                                "role": role,
                                "content": f"hi {encoding_dsv41.IMAGE_PLACEHOLDER}",
                            }
                        ],
                        thinking_mode="chat",
                    )

    def test_dsv41_tool_result_images_follow_prompt_order(self):
        """Out-of-order tool results are sorted by call order; images must follow."""
        image = encoding_dsv41.IMAGE_PLACEHOLDER
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "history.png"}},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for name in ("a", "b")
                ],
            },
        ]
        results = [
            {
                "type": "tool_result",
                "tool_use_id": name,
                "content": [
                    {"type": "text", "text": f"result {name}"},
                    {"type": "image_url", "image_url": {"url": f"{name}.png"}},
                ],
            }
            for name in ("b", "a")
        ]

        for input_format in (
            "content_blocks",
            "tool_messages",
            "tool_content_blocks",
            "context",
        ):
            with self.subTest(input_format=input_format):
                if input_format in ("tool_messages", "tool_content_blocks"):
                    content_field = (
                        "content_blocks"
                        if input_format == "tool_content_blocks"
                        else "content"
                    )
                    messages = history + [
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            content_field: block["content"],
                        }
                        for block in results
                    ]
                    context = []
                elif input_format == "context":
                    messages = [{"role": "user", "content_blocks": results}]
                    context = history
                else:
                    messages = history + [{"role": "user", "content_blocks": results}]
                    context = []
                original = copy.deepcopy((messages, context))

                prompt, media = encoding_dsv41.encode_messages(
                    messages,
                    thinking_mode="chat",
                    context=context,
                    return_multi_modal_data=True,
                )

                self.assertTrue(
                    prompt.endswith(
                        f"<｜User｜><tool_result>result a\n\n{image}</tool_result>\n\n"
                        f"<tool_result>result b\n\n{image}</tool_result>"
                        "<｜Assistant｜></think>"
                    ),
                    prompt,
                )
                expected_urls = (
                    ["a.png", "b.png"] if context else ["history.png", "a.png", "b.png"]
                )
                self.assertEqual(
                    media,
                    {"images": [{"url": url} for url in expected_urls]},
                )
                self.assertEqual((messages, context), original)

    def test_dsv41_drop_thinking_without_tools(self):
        """Tool results do not start a new turn, so current-turn reasoning is kept."""
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "reasoning_content": "r2",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "value"},
        ]
        prompt = encoding_dsv41.encode_messages(
            messages, thinking_mode="thinking", drop_thinking=True
        )
        self.assertIn(
            "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>", prompt
        )
        self.assertNotIn("r1", prompt)
        self.assertIn("<think>r2</think>", prompt)
        self.assertTrue(
            prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")
        )

        next_turn = encoding_dsv41.encode_messages(
            messages + [{"role": "user", "content": "q3"}],
            thinking_mode="thinking",
        )
        self.assertNotIn("r2", next_turn)
        self.assertTrue(next_turn.endswith("q3<｜Assistant｜><think>"))


class TestDsv41AdapterRegistry(CustomTestCase):
    def _context(self, tokenizer):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        return registry.ChatEncodingAdapterContext(
            chat_encoding_spec="dsv41",
            tokenizer=tokenizer,
            reasoning_parser="deepseek-v41",
            tool_call_parser="deepseekv41",
            chat_hooks=registry.ChatOutputHooks(
                thinking_requested=lambda request: False,
                tool_call_parsing_active=lambda request: False,
                history_tool_calls_count=lambda request: 0,
                tool_call_id=lambda item, index: "",
                effective_tools=lambda request: [],
                split_assistant_prefix=lambda messages, request: (messages, None),
                append_assistant_prefix=lambda prompt_ids, prefix: prompt_ids,
            ),
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", image_token_id=IMAGE_TOKEN_ID
            ),
            model_path="",
            revision=None,
            encode_prompt_text=lambda text: [],
        )

    def test_builds_the_in_tree_adapter_and_keeps_the_text_parsers(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.entrypoints.openai.serving_deepseek_v41 import (
            DeepSeekV41ChatEncodingAdapter,
        )

        adapter = registry.build_chat_encoding_adapter(self._context(_CharTokenizer()))
        self.assertIsInstance(adapter, DeepSeekV41ChatEncodingAdapter)
        self.assertIsNone(adapter.output)
        self.assertFalse(adapter.collects_openai_media)
        self.assertEqual(adapter.history_tool_call_arguments, "strict")
        for parser in ("deepseek-v41", "deepseekv41"):
            with self.subTest(parser=parser):
                self.assertIsNone(
                    registry.native_text_parser(
                        parser, chat_encoding_spec="dsv41", adapter=adapter
                    )
                )

    def test_no_tokenizer_builds_no_adapter(self):
        """--skip-tokenizer-init servers start, with no chat prompts to encode."""
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        self.assertIsNone(registry.build_chat_encoding_adapter(self._context(None)))

    def test_bad_env_default_fails_startup(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.environ import envs

        for raw in ("extreme", "0", "101"):
            with (
                self.subTest(raw=raw),
                envs.SGLANG_DSV41_REASONING_EFFORT.override(raw),
                self.assertRaises(ValueError),
            ):
                registry.build_chat_encoding_adapter(self._context(_CharTokenizer()))


class TestDsv41AdapterEncode(CustomTestCase):
    def test_system_message_only_hosts_request_tools(self):
        adapter = _adapter()
        hi = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            _decode(_encode(adapter, _request(hi))),
            encoding_dsv41.encode_messages(hi, thinking_mode="chat"),
        )

        tool = {
            "type": "function",
            "function": {
                "parameters": {"type": "object"},
                "description": "d",
                "name": "f",
            },
        }
        prompt = _decode(_encode(adapter, _request(hi, tools=[tool])))
        # The client's fields only, in the OpenAI order; no pydantic defaults.
        self.assertIn(
            '{"name": "f", "description": "d", "parameters": {"type": "object"}}',
            prompt,
        )
        self.assertNotIn('"strict"', prompt)
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜><｜System｜>"))

        # tool_choice "none" hides the tools from the prompt.
        prompt = adapter.encode_messages(
            messages=hi,
            request=_request(hi, tools=[tool], tool_choice="none"),
            tools=None,
            thinking=False,
            image_data=[],
        )
        self.assertNotIn("## Tools", _decode(prompt))

    def test_trailing_assistant_turn(self):
        adapter = _adapter()
        hi = {"role": "user", "content": "hi"}
        partial = {"role": "assistant", "content": "Hel"}
        # Continued: the prefix follows the generation prompt.
        self.assertEqual(
            _decode(
                _encode(adapter, _request([hi, partial], continue_final_message=True))
            ),
            encoding_dsv41.encode_messages([hi], thinking_mode="chat") + "Hel",
        )
        # Not continued: the trailing turn is re-sent as a user turn.
        self.assertEqual(
            _encode(adapter, _request([hi, partial])),
            _encode(adapter, _request([hi, {"role": "user", "content": "Hel"}])),
        )
        # A task renders its own tail; the continued prefix follows it.
        self.assertEqual(
            _decode(
                _encode(
                    adapter,
                    _request([hi, partial], continue_final_message=True, task="title"),
                )
            ),
            _decode(_encode(adapter, _request([hi], task="title"))) + "Hel",
        )

    def test_images_fill_image_data_with_the_image_token(self):
        adapter = _adapter()
        request = _request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": "a.png"}},
                    ],
                }
            ]
        )
        image_data = []
        prompt = _decode(_encode(adapter, request, image_data=image_data))
        self.assertEqual([image.url for image in image_data], ["a.png"])
        self.assertIn(IMAGE_TOKEN, prompt)
        self.assertNotIn(encoding_dsv41.IMAGE_PLACEHOLDER, prompt)

        with self.assertRaisesRegex(ValueError, "image input is not supported"):
            _encode(_adapter(image_token_id=None), request)

    def test_reasoning_effort(self):
        adapter = _adapter()
        hi = [{"role": "user", "content": "hi"}]

        def prompt(effort):
            return _decode(
                _encode(adapter, _request(hi, reasoning_effort=effort), thinking=True)
            )

        self.assertIn("Reasoning Effort: 50 ", prompt("low"))
        self.assertIn("Reasoning Effort: 50 ", prompt(0.5))
        self.assertIn("Reasoning Effort: 99 ", prompt(0.99))
        self.assertIn("Reasoning Effort: 100 ", prompt("max"))
        # Unsupported tiers fall back to the default and warn once per value.
        default = prompt(None)
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.serving_deepseek_v41", "WARNING"
        ) as logs:
            self.assertEqual(prompt("medium"), default)
            self.assertEqual(prompt("medium"), default)
            self.assertEqual(prompt("minimal"), default)
        self.assertEqual(len(logs.records), 2)


class TestDsv41Handler(CustomTestCase):
    def _serving(self):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        reset_context()
        self.addCleanup(reset_context)
        args = ServerArgs(
            model_path="dummy",
            reasoning_parser="deepseek-v41",
            tool_call_parser="deepseekv41",
        )
        publish(args, role="tokenizer")
        manager = SimpleNamespace(
            model_path="dummy",
            served_model_name="dummy",
            server_args=args,
            tokenizer=_CharTokenizer(),
            segment_encoder=None,
            model_config=SimpleNamespace(
                is_multimodal=True,
                context_len=131072,
                hf_config=SimpleNamespace(
                    model_type="deepseek_v41",
                    architectures=["DeepseekV41ForCausalLM"],
                    image_token_id=IMAGE_TOKEN_ID,
                    to_dict=lambda: {},
                ),
                get_default_sampling_params=lambda: {},
            ),
            config_value=lambda name: getattr(args, name),
        )
        template_manager = SimpleNamespace(
            chat_template_name=None,
            jinja_template_content_format=None,
            reasoning_config=None,
            force_reasoning=False,
            python_chat_template=None,
            jinja_template_may_reorder_tool_results=False,
        )
        return OpenAIServingChat(manager, template_manager)

    def test_requests_encode_through_the_adapter(self):
        from sglang.srt.entrypoints.openai.serving_deepseek_v41 import (
            DeepSeekV41ChatEncodingAdapter,
        )

        chat = self._serving()
        self.assertEqual(chat.chat_encoding_spec, "dsv41")
        self.assertIsInstance(chat.encoding_adapter, DeepSeekV41ChatEncodingAdapter)
        self.assertIsNone(chat._encoding_output())
        self.assertTrue(chat.supports_inline_system)

        messages = [{"role": "user", "content": "hi"}]
        processed = chat._process_messages(
            _request(messages, chat_template_kwargs={"thinking": True}), True
        )
        self.assertEqual(
            _decode(processed.prompt_ids),
            encoding_dsv41.encode_messages(
                messages, thinking_mode="thinking", reasoning_effort="high"
            ),
        )


if __name__ == "__main__":
    unittest.main()
