import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    GrammarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.glm4_moe_detector import (
    _GLM_XML_GRAMMAR_RULES,
    _glm_any_string_exclude,
    _glm_escape_ebnf_string,
    _glm_get_value_rule,
    _glm_has_complete_properties,
)
from sglang.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)


# Bare (suffix-less) Hunyuan special tokens. The shipping Hy3 tokenizer appends
# a shared suffix to each (e.g. ``<tool_calls:opensource>``); resolve the real
# token string from the vocab at runtime and fall back to these literals.
_HUNYUAN_TOKEN_NAMES = (
    "tool_calls",
    "tool_call",
    "tool_sep",
    "arg_key",
    "arg_value",
    "think",
)

_HUNYUAN_TOKEN_RE = re.compile(
    r"^<(?P<name>" + "|".join(_HUNYUAN_TOKEN_NAMES) + r")(?::[^>]+)?>$"
)


# id(tokenizer) -> (tokenizer, scan). The entry holds the tokenizer, so its id
# is not reused while the entry exists.
_vocab_scans: Dict[int, Tuple[Any, Optional[Dict[str, str]]]] = {}


def _scan_hunyuan_vocab(tokenizer) -> Optional[Dict[str, str]]:
    """``{name: token_str}`` for the Hunyuan tokens in the vocab, or None when
    the vocab cannot be read. Both detectors are built for every request and
    ``get_vocab()`` copies the whole vocab (~0.2 s for Hy4, on the HTTP event
    loop), so each tokenizer is scanned once."""
    entry = _vocab_scans.get(id(tokenizer))
    if entry is not None and entry[0] is tokenizer:
        return entry[1]
    found = None
    try:
        vocab = tokenizer.get_vocab()
    except Exception as e:
        logger.warning("Failed to read Hunyuan tokenizer vocab: %s", e)
        vocab = None
    if isinstance(vocab, dict):
        found = {}
        for tok in vocab:
            if not isinstance(tok, str):
                continue
            m = _HUNYUAN_TOKEN_RE.match(tok)
            if m:
                found[m.group("name")] = tok
    _vocab_scans[id(tokenizer)] = (tokenizer, found)
    return found


def resolve_hunyuan_tokens(tokenizer) -> Dict[str, str]:
    """Map bare token names to their real (possibly suffixed) strings in vocab.

    Returns ``{name: token_str}`` for each name found. A bare literal is used
    when the tokenizer carries no suffixed form, so the same detector serves
    both the preview (suffix-less) and shipping (suffixed) Hy3 tokenizers.
    """
    found = _scan_hunyuan_vocab(tokenizer) if tokenizer is not None else None
    tokens: Dict[str, str] = dict(found or {})
    for name in _HUNYUAN_TOKEN_NAMES:
        tokens.setdefault(name, f"<{name}>")
    return tokens


class HunyuanDetector(BaseFormatDetector):
    """
    Detector for Hunyuan (HYV3) tool call format.

    Format:
        <tool_calls>
        <tool_call>function_name<tool_sep>
        <arg_key>key1</arg_key>
        <arg_value>value1</arg_value>
        </tool_call>
        </tool_calls>

    Streaming behavior:
      * Phase 1 emits the tool name once <tool_sep> is seen.
      * Phase 2 streams argument JSON incrementally. Closed <arg_value>
        pairs are parsed with schema-aware type coercion; pure-string
        args may be streamed char-by-char (with JSON escaping). The
        closing "}" is withheld until </tool_call> arrives.
    """

    _TYPE_ALIASES: Dict[str, str] = {
        "str": "string",
        "text": "string",
        "varchar": "string",
        "char": "string",
        "enum": "string",
        "bool": "boolean",
        "binary": "boolean",
        "int": "integer",
        "float": "number",
        "double": "number",
        "list": "array",
        "dict": "object",
        "map": "object",
    }

    _INTEGER_PREFIXES = ("int", "uint", "long", "short", "unsigned")
    _NUMBER_PREFIXES = ("num", "float")

    def __init__(self, tokenizer=None):
        super().__init__()

        # Hy4 dropped <tool_sep> from its vocab entirely; Hy3 carries it (bare
        # or suffixed). Detect the dialect from the vocab, not from the
        # resolved literal (which falls back to "<tool_sep>" for both).
        found = _scan_hunyuan_vocab(tokenizer) if tokenizer is not None else None
        self._has_tool_sep = found is None or "tool_sep" in found

        t = resolve_hunyuan_tokens(tokenizer)
        tool_calls = t["tool_calls"]
        tool_call = t["tool_call"]
        tool_sep = t["tool_sep"]
        arg_key = t["arg_key"]
        arg_value = t["arg_value"]

        def _close(open_tok: str) -> str:
            return "</" + open_tok[1:] if open_tok.startswith("<") else open_tok

        self.bot_token = tool_calls
        self.eot_token = _close(tool_calls)
        self.tool_call_start_token = tool_call
        self.tool_call_end_token = _close(tool_call)
        self.tool_sep_token = tool_sep
        self.arg_key_start_token = arg_key
        self.arg_key_end_token = _close(arg_key)
        self.arg_value_start_token = arg_value
        self.arg_value_end_token = _close(arg_value)

        tc_end = _close(tool_call)
        ak_end = _close(arg_key)
        av_end = _close(arg_value)
        self.tool_call_regex = re.compile(
            re.escape(tool_call)
            + r"(.*?)"
            + r"(?:"
            + re.escape(tool_sep)
            + r"|(?="
            + re.escape(arg_key)
            + r"|"
            + re.escape(tc_end)
            + r"))"
            + r"(.*?)"
            + re.escape(tc_end),
            re.DOTALL,
        )
        self.func_args_regex = re.compile(
            re.escape(arg_key)
            + r"(.*?)"
            + re.escape(ak_end)
            + r"\s*"
            + re.escape(arg_value)
            + r"(.*?)"
            + re.escape(av_end),
            re.DOTALL,
        )

        # Streaming state
        self._in_tool_calls: bool = False
        self._streaming_tool_name: Optional[str] = None
        self._completed_args: Dict[str, Any] = {}
        self._streamed_json_len: int = 0

    # ------------------------------------------------------------------
    # Type-normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        exact = HunyuanDetector._TYPE_ALIASES.get(raw_type)
        if exact is not None:
            return exact
        lower = raw_type.lower()
        if any(lower.startswith(p) for p in HunyuanDetector._INTEGER_PREFIXES):
            return "integer"
        if any(lower.startswith(p) for p in HunyuanDetector._NUMBER_PREFIXES):
            return "number"
        return raw_type

    @staticmethod
    def _get_arg_schema(
        function_name: str, arg_key: str, tools: Optional[List[Tool]]
    ) -> dict:
        if not tools:
            return {}
        for tool in tools:
            if tool.function.name == function_name:
                if tool.function.parameters is None:
                    return {}
                return get_schema_properties(tool.function.parameters).get(arg_key, {})
        return {}

    @staticmethod
    def _get_schema_options(arg_schema: dict) -> List[dict]:
        """Priority: single ``type`` > ``anyOf`` > ``oneOf``; else default string."""
        if "type" in arg_schema:
            return [arg_schema]
        if "anyOf" in arg_schema:
            return arg_schema["anyOf"]
        if "oneOf" in arg_schema:
            return arg_schema["oneOf"]
        return [{"type": "string"}]

    @staticmethod
    def _get_types(arg_schema: dict) -> Set[str]:
        schemas = HunyuanDetector._get_schema_options(arg_schema)
        return {
            HunyuanDetector._normalize_type(s.get("type", "string")) for s in schemas
        } - {"null"}

    @staticmethod
    def _is_only_string_type(
        function_name: str, arg_key: str, tools: Optional[List[Tool]]
    ) -> bool:
        """Only pure-string args get char-by-char value streaming; compound
        types like anyOf(string | array) might resolve to a JSON array or
        object, so we can't safely stream them as open JSON strings."""
        arg_schema = HunyuanDetector._get_arg_schema(function_name, arg_key, tools)
        return HunyuanDetector._get_types(arg_schema) == {"string"}

    @staticmethod
    def _try_parse_bool(value: str) -> Optional[bool]:
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        return None

    @staticmethod
    def _try_parse_int(value: str) -> Optional[int]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _try_parse_number(value: str):
        """int if no '.'/'e'/'E', else float."""
        try:
            if "." in value or "e" in value or "E" in value:
                return float(value)
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    @staticmethod
    def _parse_value(
        value: str,
        function_name: str,
        arg_key: str,
        tools: Optional[List[Tool]],
    ) -> Any:
        """Unified value parser: bool → int → number → json (array/obj) → string."""
        arg_schema = HunyuanDetector._get_arg_schema(function_name, arg_key, tools)
        types = HunyuanDetector._get_types(arg_schema)

        if "boolean" in types:
            r = HunyuanDetector._try_parse_bool(value)
            if r is not None:
                return r

        if "integer" in types:
            r = HunyuanDetector._try_parse_int(value)
            if r is not None:
                return r

        if "number" in types:
            r = HunyuanDetector._try_parse_number(value)
            if r is not None:
                return r

        if types - {"string", "boolean", "integer", "number"}:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass

        if "string" in types:
            return value

        return HunyuanDetector._deserialize(value)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx > 0 else ""

        tool_indices = self._get_tool_indices(tools)
        forward_unknown = envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()

        calls: List[ToolCallItem] = []
        try:
            for function_name, function_args in self.tool_call_regex.findall(text):
                function_name = function_name.strip()
                if function_name not in tool_indices and not forward_unknown:
                    logger.warning(
                        "Model attempted to call undefined function: %s", function_name
                    )
                    continue

                arg_dict: Dict[str, Any] = {}
                for key, value in self.func_args_regex.findall(function_args):
                    key = key.strip()
                    arg_dict[key] = self._parse_value(value, function_name, key, tools)

                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(function_name, -1),
                        name=function_name,
                        parameters=json.dumps(arg_dict, ensure_ascii=False),
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}", exc_info=True)
            return StreamingParseResult(normal_text=text)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _reset_streaming_tool_state(self):
        self._streaming_tool_name = None
        self._completed_args = {}
        self._streamed_json_len = 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        try:
            return self._parse_streaming_increment_impl(new_text, tools)
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            return StreamingParseResult()

    def _parse_streaming_increment_impl(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Not yet inside <tool_calls>: emit normal text or buffer partial bot_token.
        if not self._in_tool_calls:
            combined = self._buffer + new_text
            if self.bot_token in combined:
                bot_pos = combined.find(self.bot_token)
                normal_text = combined[:bot_pos]
                self._buffer = combined[bot_pos + len(self.bot_token) :]
                self._in_tool_calls = True
                return self._continue_streaming(tools, leading_normal=normal_text)

            partial_len = self._ends_with_partial_token(combined, self.bot_token)
            if partial_len:
                self._buffer = combined[-partial_len:]
                return StreamingParseResult(normal_text=combined[:-partial_len])
            self._buffer = ""
            return StreamingParseResult(normal_text=combined)

        self._buffer += new_text
        return self._continue_streaming(tools)

    def _continue_streaming(
        self, tools: List[Tool], leading_normal: str = ""
    ) -> StreamingParseResult:
        """Drive the state machine after <tool_calls> is open."""
        calls: List[ToolCallItem] = []

        while True:
            if self._streaming_tool_name is None:
                # Phase 1: wait for <tool_call>..<tool_sep>.
                tc_start = self._buffer.find(self.tool_call_start_token)
                if tc_start == -1:
                    if self.eot_token in self._buffer:
                        eot_pos = self._buffer.find(self.eot_token)
                        self._buffer = self._buffer[eot_pos + len(self.eot_token) :]
                        self._in_tool_calls = False
                    break

                name_start = tc_start + len(self.tool_call_start_token)
                boundaries = [
                    (self._buffer.find(token, name_start), token)
                    for token in (
                        self.tool_sep_token,
                        self.arg_key_start_token,
                        self.tool_call_end_token,
                    )
                ]
                boundaries = [(pos, token) for pos, token in boundaries if pos != -1]
                if not boundaries:
                    self._buffer = self._buffer[tc_start:]
                    break
                boundary_pos, boundary_token = min(boundaries, key=lambda item: item[0])

                tool_name = self._buffer[name_start:boundary_pos].strip()

                if (
                    tool_name not in self._tool_indices
                    and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    logger.warning(
                        "Model attempted to call undefined function: %s", tool_name
                    )

                self._streaming_tool_name = tool_name
                self.current_tool_id += 1
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=tool_name,
                        parameters="",
                    )
                )

                if boundary_token == self.tool_sep_token:
                    boundary_pos += len(boundary_token)
                self._buffer = self._buffer[boundary_pos:]

            # Phase 2: stream argument JSON of the current tool.
            before_name = self._streaming_tool_name
            calls.extend(self._stream_args(tools))
            if self._streaming_tool_name is not None:
                break  # current tool still open; need more data.
            if self._streaming_tool_name == before_name:
                break  # safety: avoid infinite loop if state didn't advance.

        return StreamingParseResult(normal_text=leading_normal, calls=calls)

    def _stream_args(self, tools: List[Tool]) -> List[ToolCallItem]:
        """Emit argument-JSON deltas for the currently-open tool call."""
        is_complete = self.tool_call_end_token in self._buffer

        if is_complete:
            end_idx = self._buffer.find(self.tool_call_end_token)
            args_text = self._buffer[:end_idx]
        else:
            args_text = self._buffer

        # 1. Absorb closed <arg_key>..<arg_value> pairs.
        last_closed_end = 0
        for m in self.func_args_regex.finditer(args_text):
            key, value = m.groups()
            key = key.strip()
            if key not in self._completed_args:
                self._completed_args[key] = self._parse_value(
                    value, self._streaming_tool_name or "", key, tools
                )
            last_closed_end = m.end()

        # 2. Detect a partial (unclosed) kv pair at the tail.
        tail = args_text[last_closed_end:]
        partial_key: Optional[str] = None
        partial_value: Optional[str] = None

        ak_start = tail.find(self.arg_key_start_token)
        if ak_start != -1:
            ak_end = tail.find(
                self.arg_key_end_token, ak_start + len(self.arg_key_start_token)
            )
            if ak_end != -1:
                partial_key = tail[
                    ak_start + len(self.arg_key_start_token) : ak_end
                ].strip()
                av_start = tail.find(self.arg_value_start_token, ak_end)
                if av_start != -1 and self._is_only_string_type(
                    self._streaming_tool_name or "", partial_key, tools
                ):
                    partial_value = tail[av_start + len(self.arg_value_start_token) :]

        # Avoid emitting a lone "{" before any arg content is knowable.
        if not is_complete and not self._completed_args and partial_value is None:
            return []

        # 3. Build the JSON snapshot manually to control streaming boundaries.
        snapshot_parts: List[str] = []
        for k, v in self._completed_args.items():
            k_json = json.dumps(k, ensure_ascii=False)
            v_json = json.dumps(v, ensure_ascii=False)
            snapshot_parts.append(f"{k_json}: {v_json}")

        if partial_key is not None and partial_value is not None:
            # Hold back chars that could be a partial </arg_value> marker so
            # that a `<` starting the end-tag doesn't leak into the streamed
            # JSON string value.
            hold = self._ends_with_partial_token(
                partial_value, self.arg_value_end_token
            )
            safe_value = partial_value[:-hold] if hold else partial_value
            k_json = json.dumps(partial_key, ensure_ascii=False)
            escaped = (
                safe_value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            # No closing `"` here — it's appended when the value closes.
            snapshot_parts.append(f'{k_json}: "{escaped}')

        snapshot = "{" + ", ".join(snapshot_parts) + "}"

        argument_diff: Optional[str] = None

        if is_complete:
            final_json = json.dumps(self._completed_args, ensure_ascii=False)
            if self._streamed_json_len < len(final_json):
                argument_diff = final_json[self._streamed_json_len :]
            self._streamed_json_len = len(final_json)

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": self._streaming_tool_name,
                "arguments": dict(self._completed_args),
            }

            end_idx = self._buffer.find(self.tool_call_end_token)
            self._buffer = self._buffer[end_idx + len(self.tool_call_end_token) :]
            self._reset_streaming_tool_state()
        else:
            # Withhold the trailing "}" while the tool call is still open.
            end = len(snapshot) - 1
            if end > self._streamed_json_len:
                argument_diff = snapshot[self._streamed_json_len : end]
                self._streamed_json_len = end

        if argument_diff:
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            return [
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    parameters=argument_diff,
                )
            ]
        return []

    def structure_info(self) -> _GetInfoFunc:
        # Hy4 has no <tool_sep> and no separators between adjacent
        # <tool_call> blocks, so the Hy3 begin/end literals never match.
        if not self._has_tool_sep:
            return lambda name: StructureInfo(
                begin=f"{self.bot_token}{self.tool_call_start_token}{name}",
                end=f"{self.tool_call_end_token}{self.eot_token}",
                trigger=self.bot_token,
            )
        return lambda name: StructureInfo(
            begin=f"{self.bot_token}\n{self.tool_call_start_token}{name}{self.tool_sep_token}",
            end=f"{self.tool_call_end_token}\n{self.eot_token}",
            trigger=self.bot_token,
        )

    def supports_structural_tag(self) -> bool:
        return False

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        # Hy4 only (the Hy3 dialect puts <tool_sep> and newlines between the
        # parts). With thinking_mode the tag would also have to cover the
        # reasoning; Hy4 serves with a reasoning parser, which owns it.
        if self._has_tool_sep or thinking_mode or not tools:
            return None
        if isinstance(tool_choice, ToolChoice):
            tools = [t for t in tools if t.function.name == tool_choice.function.name]
            if not tools:
                return None
        calls = TagFormat(
            begin=self.bot_token,
            content=TagsWithSeparatorFormat(
                tags=[
                    TagFormat(
                        begin=f"{self.tool_call_start_token}{tool.function.name}",
                        content=GrammarFormat(
                            grammar=self._arguments_grammar(tool.function.parameters)
                        ),
                        end=self.tool_call_end_token,
                    )
                    for tool in tools
                ],
                separator="",
                at_least_one=True,
                stop_after_first=not parallel_tool_calls,
            ),
            end=self.eot_token,
        )
        if tool_choice == "auto":
            return StructuralTag(
                format=TriggeredTagsFormat(
                    triggers=[self.bot_token], tags=[calls], stop_after_first=True
                )
            )
        return StructuralTag(format=calls)

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        # Free text and any number of calls need no grammar; only a one-call
        # limit does.
        if parallel_tool_calls:
            return None
        return self.get_structural_tag(
            tools=tools,
            tool_choice="auto",
            thinking_mode=thinking_mode,
            parallel_tool_calls=False,
        )

    def _arguments_grammar(self, parameters: Optional[dict]) -> str:
        """EBNF for the argument pairs of one call: the GLM-4.7 rules with Hy4's
        tags. Keys come from the schema, in any order; values are typed for
        enums and scalar types, free text otherwise."""
        special_tokens = [
            self.bot_token,
            self.eot_token,
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.arg_key_start_token,
            self.arg_key_end_token,
            self.arg_value_start_token,
            self.arg_value_end_token,
        ]
        params = parameters or {}
        properties = (
            get_schema_properties(params) if _glm_has_complete_properties(params) else {}
        )
        composed = any(k in params for k in ("allOf", "anyOf", "oneOf"))

        def pair(key_rule: str, value_rule: str) -> str:
            return (
                f'"{self.arg_key_start_token}" {key_rule} "{self.arg_key_end_token}" '
                f'"{self.arg_value_start_token}" ({value_rule}) "{self.arg_value_end_token}"'
            )

        if properties:
            choices = " | ".join(
                pair(
                    f'"{_glm_escape_ebnf_string(name)}"',
                    "text_without_special_tokens"
                    if composed
                    else _glm_get_value_rule(schema),
                )
                for name, schema in properties.items()
            )
        else:
            choices = pair("text_without_special_tokens", "text_without_special_tokens")
        return "\n".join(
            [
                f"root ::= ( {choices} )*",
                *_glm_any_string_exclude("text_without_special_tokens", special_tokens),
                *_GLM_XML_GRAMMAR_RULES,
            ]
        )

    def parses_required_natively(self) -> bool:
        # Hy3/Hy4 both emit their structural tool-call format natively; without
        # a dialect grammar, a json_schema constraint would force plain JSON the
        # model was not trained to produce here.
        return True
