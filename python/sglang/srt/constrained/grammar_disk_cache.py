"""Compiled xgrammar grammars shared through a directory.

Every TP rank compiles the same grammar for a request. A tool-call grammar such as
GLM-5.3's takes about 10 CPU-seconds, and eight ranks compiling it at once on the
GPUs' NUMA node delay the request by several seconds. With this cache the first rank
to lock an entry compiles and writes it, and the other ranks load the file. Entries
outlive the server, so a restart with the same tools skips the compile.
"""

import fcntl
import hashlib
import importlib.metadata
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from xgrammar import CompiledGrammar, TokenizerInfo

logger = logging.getLogger(__name__)


class GrammarDiskCache:
    def __init__(
        self, directory: str, tokenizer_info: TokenizerInfo, max_entries: int
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer_info = tokenizer_info
        self.max_entries = max_entries
        # A serialized grammar is only valid for the xgrammar version and the
        # vocabulary it was compiled with.
        fingerprint = hashlib.sha256()
        fingerprint.update(importlib.metadata.version("xgrammar").encode())
        fingerprint.update(tokenizer_info.serialize_json().encode())
        self._fingerprint = fingerprint.hexdigest()
        self._write_failed = False

    def entry_path(self, kind: str, key: str) -> Path:
        digest = hashlib.sha256(
            f"{self._fingerprint}\0{kind}\0{key}".encode()
        ).hexdigest()
        return self.directory / f"{digest}.json"

    def get_or_compile(
        self, kind: str, key: str, compile_fn: Callable[[], CompiledGrammar]
    ) -> CompiledGrammar:
        if self._write_failed:
            # Without writes, waiting on the lock would serialize the compiles.
            return compile_fn()
        path = self.entry_path(kind, key)
        compiled = self._load(path)
        if compiled is not None:
            return compiled
        # flock is released when the holder exits, so a crashed rank cannot
        # leave the others waiting. Waiters load after unlocking, side by side.
        with open(path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if not path.exists():
                    return self._compile_and_store(path, compile_fn)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        compiled = self._load(path)
        if compiled is None:
            compiled = self._compile_and_store(path, compile_fn)
        return compiled

    def _compile_and_store(
        self, path: Path, compile_fn: Callable[[], CompiledGrammar]
    ) -> CompiledGrammar:
        start = time.perf_counter()
        compiled = compile_fn()
        compiled_at = time.perf_counter()
        self._store(path, compiled)
        logger.info(
            "Grammar compiled in %.2f s, cached in %.2f s: %s",
            compiled_at - start,
            time.perf_counter() - compiled_at,
            path.name,
        )
        return compiled

    def _load(self, path: Path) -> Optional[CompiledGrammar]:
        start = time.perf_counter()
        try:
            data = path.read_text()
        except FileNotFoundError:
            return None
        try:
            compiled = CompiledGrammar.deserialize_json(data, self.tokenizer_info)
        except Exception as e:
            logger.warning("Ignoring unreadable grammar cache entry %s: %s", path, e)
            return None
        try:
            os.utime(path)
        except OSError:
            pass
        logger.info(
            "Grammar loaded in %.2f s: %s", time.perf_counter() - start, path.name
        )
        return compiled

    def _store(self, path: Path, compiled: CompiledGrammar) -> None:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        try:
            tmp.write_text(compiled.serialize_json())
            os.replace(tmp, path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            if not self._write_failed:
                logger.warning("Cannot write grammar cache entry %s: %s", path, e)
                self._write_failed = True
            return
        self._evict()

    def _evict(self) -> None:
        entries = []
        for entry in self.directory.glob("*.json"):
            try:
                entries.append((entry.stat().st_mtime, entry))
            except FileNotFoundError:
                continue
        entries.sort()
        for _, entry in entries[: max(0, len(entries) - self.max_entries)]:
            entry.unlink(missing_ok=True)
            entry.with_suffix(".lock").unlink(missing_ok=True)
