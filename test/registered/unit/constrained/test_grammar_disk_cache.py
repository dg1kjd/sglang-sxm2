"""
Unit tests for sglang.srt.constrained.grammar_disk_cache.

Test Coverage:
- A second backend (another TP rank, or the server after a restart) loads the
  grammar from disk instead of compiling it, and the loaded grammar masks the
  same tokens as the compiled one
- Concurrent ranks compile an entry once
- Unreadable entries are recompiled, old entries are evicted
- Without SGLANG_GRAMMAR_DISK_CACHE_DIR the backend has no disk cache

Usage:
    python -m pytest test_grammar_disk_cache.py -v
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from xgrammar import GrammarCompiler, TokenizerInfo

from sglang.srt.constrained.grammar_disk_cache import GrammarDiskCache
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")

EOS = 128
EBNF = 'root ::= "call(" [a-z]+ ")" | "no"'


def _tokenizer_info():
    vocab = [chr(token_id) for token_id in range(128)] + ["<eos>"]
    return TokenizerInfo(vocab, stop_token_ids=[EOS])


def _backend(info):
    tokenizer = SimpleNamespace(init_xgrammar=lambda: (info, None))
    return XGrammarGrammarBackend(tokenizer, EOS + 1)


def _masks(grammar, text):
    """Vocab masks at each step of accepting ``text``, then at the end."""
    masks = []
    for token_id in [ord(c) for c in text]:
        mask = torch.zeros((1, (EOS + 1 + 31) // 32), dtype=torch.int32)
        grammar.fill_vocab_mask(mask, 0)
        masks.append(mask)
        grammar.accept_token(token_id)
    mask = torch.zeros((1, (EOS + 1 + 31) // 32), dtype=torch.int32)
    grammar.fill_vocab_mask(mask, 0)
    masks.append(mask)
    return torch.cat(masks)


class TestGrammarDiskCache(CustomTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        self.info = _tokenizer_info()

    def tearDown(self):
        self._dir.cleanup()

    def test_second_backend_loads_instead_of_compiling(self):
        with envs.SGLANG_GRAMMAR_DISK_CACHE_DIR.override(str(self.dir)):
            first = _backend(self.info)
            compiled = first.dispatch_ebnf(EBNF)
            self.assertEqual(len(list(self.dir.glob("*.json"))), 1)
            second = _backend(self.info)
            with patch.object(
                second.grammar_compiler,
                "compile_grammar",
                side_effect=AssertionError("compiled despite a cache entry"),
            ):
                loaded = second.dispatch_ebnf(EBNF)
        self.assertTrue(
            torch.equal(_masks(compiled, "call(ab)"), _masks(loaded, "call(ab)"))
        )

    def test_concurrent_ranks_compile_once(self):
        calls = []
        compiler = GrammarCompiler(self.info, cache_enabled=False)

        def compile_fn():
            calls.append(1)
            time.sleep(0.2)
            return compiler.compile_grammar(EBNF)

        results = [None] * 4

        def rank(i):
            cache = GrammarDiskCache(str(self.dir), self.info, max_entries=8)
            results[i] = cache.get_or_compile("ebnf", EBNF, compile_fn)

        threads = [threading.Thread(target=rank, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(r is not None for r in results))

    def test_unreadable_entry_is_recompiled(self):
        cache = GrammarDiskCache(str(self.dir), self.info, max_entries=8)
        path = cache.entry_path("ebnf", EBNF)
        path.write_text("not a grammar")
        compiler = GrammarCompiler(self.info, cache_enabled=False)
        calls = []

        def compile_fn():
            calls.append(1)
            return compiler.compile_grammar(EBNF)

        cache.get_or_compile("ebnf", EBNF, compile_fn)
        cache.get_or_compile("ebnf", EBNF, compile_fn)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(path.read_text(), "not a grammar")

    def test_oldest_entries_are_evicted(self):
        cache = GrammarDiskCache(str(self.dir), self.info, max_entries=2)
        compiler = GrammarCompiler(self.info, cache_enabled=False)
        keys = ['root ::= "a"', 'root ::= "b"', 'root ::= "c"']
        for key in keys:
            cache.get_or_compile("ebnf", key, lambda: compiler.compile_grammar(key))
        remaining = {p.name for p in self.dir.glob("*.json")}
        self.assertEqual(
            remaining, {cache.entry_path("ebnf", k).name for k in keys[1:]}
        )

    def test_vocabulary_is_part_of_the_key(self):
        other_info = TokenizerInfo(
            [chr(token_id) for token_id in range(128)] + ["<end>"],
            stop_token_ids=[EOS],
        )
        cache = GrammarDiskCache(str(self.dir), self.info, max_entries=8)
        other = GrammarDiskCache(str(self.dir), other_info, max_entries=8)
        self.assertNotEqual(
            cache.entry_path("ebnf", EBNF), other.entry_path("ebnf", EBNF)
        )

    def test_unset_directory_disables_the_cache(self):
        with envs.SGLANG_GRAMMAR_DISK_CACHE_DIR.override(None):
            backend = _backend(self.info)
        self.assertIsNone(backend._disk_cache)


if __name__ == "__main__":
    unittest.main()
