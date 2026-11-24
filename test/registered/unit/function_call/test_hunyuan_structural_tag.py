"""Hy4 tool_choice constraints: required, named and parallel_tool_calls=false."""

import unittest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

S = ":opensource>"
CALLS, CALLS_END = "<tool_calls" + S, "</tool_calls" + S
CALL, CALL_END = "<tool_call" + S, "</tool_call" + S
KEY, KEY_END = "<arg_key" + S, "</arg_key" + S
VAL, VAL_END = "<arg_value" + S, "</arg_value" + S


class _Hy4Tokenizer:
    def get_vocab(self):
        return {
            CALLS[:-1] + ">": 1,
            CALL[:-1] + ">": 2,
            KEY[:-1] + ">": 3,
            VAL[:-1] + ">": 4,
            "<think" + S: 5,
        }


class _Hy3Tokenizer:
    def get_vocab(self):
        return {"<tool_calls>": 1, "<tool_call>": 2, "<tool_sep>": 3}


def _tools():
    weather = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    }
    return [
        Tool(type="function", function=Function(name="get_weather", parameters=weather)),
        Tool(type="function", function=Function(name="get_time", parameters={})),
    ]


def _call(name, **args):
    pairs = "".join(f"{KEY}{k}{KEY_END}{VAL}{v}{VAL_END}" for k, v in args.items())
    return f"{CALL}{name}{pairs}{CALL_END}"


def _grammar(tool_choice, parallel_tool_calls=True, tokenizer=None):
    parser = FunctionCallParser(
        _tools(), "hunyuan", tokenizer=tokenizer or _Hy4Tokenizer()
    )
    constraint = parser.get_structure_constraint(
        tool_choice, parallel_tool_calls=parallel_tool_calls
    )
    if constraint is None:
        return None
    assert constraint[0] == "structural_tag", constraint[0]
    return xgr.Grammar.from_structural_tag(constraint[1])


class TestHy4ToolChoiceStructuralTag(CustomTestCase):
    def test_required_forces_a_call(self):
        g = _grammar("required")
        self.assertIsNotNone(g)
        one = CALLS + _call("get_weather", city="Paris", unit="c") + CALLS_END
        self.assertTrue(_is_grammar_accept_string(g, one))
        self.assertTrue(_is_grammar_accept_string(g, CALLS + _call("get_time") + CALLS_END))
        two = CALLS + _call("get_weather", city="Paris") + _call("get_time") + CALLS_END
        self.assertTrue(_is_grammar_accept_string(g, two))
        self.assertFalse(_is_grammar_accept_string(g, "Hello!"))
        self.assertFalse(_is_grammar_accept_string(g, CALLS + CALLS_END))
        self.assertFalse(_is_grammar_accept_string(g, CALLS + _call("nope") + CALLS_END))

    def test_argument_keys_and_values_follow_the_schema(self):
        g = _grammar("required")
        self.assertIsNotNone(g)
        bad_key = CALLS + _call("get_weather", town="Paris") + CALLS_END
        bad_enum = CALLS + _call("get_weather", city="Paris", unit="k") + CALLS_END
        bad_int = CALLS + _call("get_weather", city="Paris", days="two") + CALLS_END
        good_int = CALLS + _call("get_weather", city="Paris", days="2") + CALLS_END
        self.assertFalse(_is_grammar_accept_string(g, bad_key))
        self.assertFalse(_is_grammar_accept_string(g, bad_enum))
        self.assertFalse(_is_grammar_accept_string(g, bad_int))
        self.assertTrue(_is_grammar_accept_string(g, good_int))

    def test_named_tool_choice_allows_only_that_tool(self):
        g = _grammar(ToolChoice(function=ToolChoiceFuncName(name="get_time")))
        self.assertIsNotNone(g)
        self.assertTrue(_is_grammar_accept_string(g, CALLS + _call("get_time") + CALLS_END))
        other = CALLS + _call("get_weather", city="Paris") + CALLS_END
        self.assertFalse(_is_grammar_accept_string(g, other))

    def test_parallel_off_allows_one_call(self):
        g = _grammar("required", parallel_tool_calls=False)
        self.assertIsNotNone(g)
        one = CALLS + _call("get_weather", city="Paris") + CALLS_END
        two = CALLS + _call("get_weather", city="Paris") + _call("get_time") + CALLS_END
        self.assertTrue(_is_grammar_accept_string(g, one))
        self.assertFalse(_is_grammar_accept_string(g, two))

    def test_auto_parallel_off_allows_text_or_one_call(self):
        g = _grammar("auto", parallel_tool_calls=False)
        self.assertIsNotNone(g)
        self.assertTrue(_is_grammar_accept_string(g, "No tool needed."))
        one = "Checking." + CALLS + _call("get_time") + CALLS_END
        two = CALLS + _call("get_time") + _call("get_time") + CALLS_END
        self.assertTrue(_is_grammar_accept_string(g, one))
        self.assertFalse(_is_grammar_accept_string(g, two))

    def test_auto_parallel_on_stays_unconstrained(self):
        self.assertIsNone(_grammar("auto"))

    def test_hy3_dialect_unchanged(self):
        self.assertIsNone(_grammar("required", tokenizer=_Hy3Tokenizer()))


if __name__ == "__main__":
    unittest.main()
