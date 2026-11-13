"""The deepseek-v41 reasoning parser: the V100 port keeps DeepSeek-V4.1 output on
the text parsers, so a calls block that starts a line ends the reasoning."""

import unittest

from sglang.srt.parser.reasoning_parser import (
    DeepSeekV41ReasoningDetector,
    ReasoningParser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DSML = "｜DSML｜"
CALLS = f"<{DSML} calls>"
TOOL_CALL = (
    f"{CALLS}\n"
    f'<{DSML} invoke name="s">\n'
    f'<{DSML} parameter name="q" string="true">x</{DSML} parameter>\n'
    f"</{DSML} invoke>\n"
    f"</{DSML} calls>"
)
CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, 1000]


def _feed(detector, text, chunk_size):
    reasoning = normal = ""
    for i in range(0, len(text), chunk_size):
        result = detector.parse_streaming_increment(text[i : i + chunk_size])
        reasoning += result.reasoning_text or ""
        normal += result.normal_text or ""
    result = detector.finish()
    return reasoning + (result.reasoning_text or ""), normal + (
        result.normal_text or ""
    )


class TestDeepSeekV41Reasoning(CustomTestCase):
    def _assert_invariant(self, text, expected, **kwargs):
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    _feed(DeepSeekV41ReasoningDetector(**kwargs), text, chunk_size),
                    expected,
                )
        one_shot = DeepSeekV41ReasoningDetector(**kwargs).detect_and_parse(text)
        self.assertEqual(
            (one_shot.reasoning_text or "", one_shot.normal_text or ""), expected
        )

    def test_registered_under_its_parser_name(self):
        detector = ReasoningParser(model_type="deepseek-v41").detector
        self.assertIsInstance(detector, DeepSeekV41ReasoningDetector)
        self.assertEqual(detector.tool_start_token, CALLS)
        self.assertTrue(detector.tool_start_at_line_start)

    def test_calls_block_at_line_start_ends_reasoning(self):
        """Without </think>, a calls block that starts a line still reaches
        normal_text, where the tool-call parser sees it."""
        self._assert_invariant(
            f"<think>pick a tool\n\n{TOOL_CALL}", ("pick a tool\n\n", TOOL_CALL)
        )

    def test_calls_block_after_think_end(self):
        self._assert_invariant(
            f"<think>my reasoning</think>\n\n{TOOL_CALL}",
            ("my reasoning", f"\n\n{TOOL_CALL}"),
        )

    def test_mid_sentence_mention_stays_in_reasoning(self):
        """The system prompt shows the model the calls tag, so it may name it
        while thinking; a mention that does not start a line is not a call."""
        self._assert_invariant(
            f"<think>I'll output a {CALLS} block.</think>done",
            (f"I'll output a {CALLS} block.", "done"),
        )

    def test_unspaced_v4_tag_does_not_end_reasoning(self):
        v4_block = f"<{DSML}tool_calls>"
        self._assert_invariant(
            f"<think>a\n{v4_block}</think>b", (f"a\n{v4_block}", "b")
        )

    def test_truncated_calls_opener_is_not_flushed_into_reasoning(self):
        """A generation cut off inside the calls opener (max_tokens before `>`)
        must not surface the partial tag in reasoning_content."""
        text = f"Choosing a tool\n\n{CALLS[:-1]}"
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(
                    _feed(
                        DeepSeekV41ReasoningDetector(force_reasoning=True),
                        text,
                        chunk_size,
                    ),
                    ("Choosing a tool\n\n", ""),
                )
        result = DeepSeekV41ReasoningDetector(force_reasoning=True).detect_and_parse(
            text
        )
        self.assertEqual(result.reasoning_text, "Choosing a tool\n\n")
        self.assertFalse(result.normal_text)

    def test_mid_sentence_partial_opener_survives_the_flush(self):
        text = f"mentioning {CALLS[:-3]}"
        self.assertEqual(
            _feed(DeepSeekV41ReasoningDetector(force_reasoning=True), text, 1000),
            (text, ""),
        )


if __name__ == "__main__":
    unittest.main()
