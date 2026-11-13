"""Read only the checkpoint tensors one DSV4.1 rank actually copies.

The on-disk checkpoint is one 475 GiB safetensors set. Every TP/EP rank used
to materialize every tensor and then drop it. Routed experts are a contiguous
EP split (about 7/8 of 276 GiB thrown away per rank). Engram embed tables are
two ~90 GiB tensors, and each rank keeps one TP row-slice. The DSpark draft
only consumes ``mtp.*``.

``DSV4CheckpointReader`` returns None for a tensor the rank will not copy, and
returns an already-sliced Engram shard so ``EngramEmbedding._load_rows`` does
not need the full table. The model passes each table's row range, taken from
its ``EngramEmbedding``, so the reader never computes a split of its own. It
records which expert ids and Engram tables it yielded. ``finish`` raises if an
id this rank owns or a table it has a range for never appeared, so a bad filter
fails the load instead of leaving a weight at its empty init value.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import FrozenSet, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

_EXPERT_ID_IN_NAME = re.compile(r"\.ffn\.experts\.(\d+)\.")
_ENGRAM_EMBED = re.compile(r"\.engram\.embed\.(weight|scale)$")


def expert_id_in_checkpoint_name(name: str) -> Optional[int]:
    match = _EXPERT_ID_IN_NAME.search(name)
    if match is None:
        return None
    return int(match.group(1))


def is_engram_embed_table(name: str) -> bool:
    return _ENGRAM_EMBED.search(name) is not None


def contiguous_owned_expert_ids(
    n_routed: int, ep_rank: int, ep_size: int
) -> Optional[FrozenSet[int]]:
    """Logical expert ids of a plain EP split. None when the split is not even."""
    if ep_size <= 1 or n_routed <= 0 or ep_rank < 0 or ep_rank >= ep_size:
        return None
    if n_routed % ep_size != 0:
        return None
    local = n_routed // ep_size
    start = ep_rank * local
    return frozenset(range(start, start + local))


def owned_expert_ids_from_physical_map(
    physical_to_logical: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    n_routed: int,
) -> Optional[FrozenSet[int]]:
    """Logical ids stored in this rank's physical slots, across every layer.

    The union is a superset of what any single layer needs. An unusable map
    returns None, and the caller must not filter.
    """
    if ep_size <= 0 or ep_rank < 0 or ep_rank >= ep_size or n_routed <= 0:
        return None
    if physical_to_logical.ndim != 2:
        return None
    num_physical = int(physical_to_logical.shape[1])
    if num_physical % ep_size != 0:
        return None
    local = num_physical // ep_size
    start = ep_rank * local
    end = start + local
    values = physical_to_logical[:, start:end].reshape(-1).tolist()
    owned = frozenset(int(v) for v in values if 0 <= int(v) < n_routed)
    if not owned:
        return None
    return owned


def engram_table_prefix(name: str) -> str:
    """``layers.N.engram.embed.weight`` -> ``layers.N.engram.embed.``"""
    return name[: name.rindex(".") + 1]


class DSV4CheckpointReader:
    """Callable ``(name, safe_open_handle) -> tensor or None``.

    ``engram_rows`` maps an Engram table's checkpoint prefix
    (``layers.N.engram.embed.``) to this rank's ``[start, end)`` rows; None
    reads the tables whole.
    """

    def __init__(
        self,
        *,
        owned_expert_ids: Optional[FrozenSet[int]],
        n_routed: int,
        mtp_only: bool,
        engram_rows: Optional[Mapping[str, tuple[int, int]]],
    ) -> None:
        self.owned_expert_ids = owned_expert_ids
        self.n_routed = int(n_routed)
        self.mtp_only = bool(mtp_only)
        self.engram_rows = None if engram_rows is None else dict(engram_rows)
        self.read = 0
        self.skipped = 0
        self.expert_ids_read: set[int] = set()
        self.engram_tables_read: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, name: str, handle) -> Optional[torch.Tensor]:
        if self.mtp_only and not name.startswith("mtp."):
            with self._lock:
                self.skipped += 1
            return None
        expert_id = expert_id_in_checkpoint_name(name)
        if (
            expert_id is not None
            and self.owned_expert_ids is not None
            and expert_id < self.n_routed
            and expert_id not in self.owned_expert_ids
        ):
            with self._lock:
                self.skipped += 1
            return None
        engram_prefix = None
        if self.engram_rows is not None and is_engram_embed_table(name):
            engram_prefix = engram_table_prefix(name)
            if engram_prefix not in self.engram_rows:
                raise RuntimeError(
                    f"DSV4 checkpoint read: no Engram row range for {name}"
                )
            row_start, row_end = self.engram_rows[engram_prefix]
            view = handle.get_slice(name)
            # A row slice is a view onto the mmap: its storage size is the
            # whole table, but reads and copy_ touch only these rows, and the
            # view stays valid after safe_open returns. Do not clone. Each
            # table is ~90 GiB and a clone is a private copy of this rank's
            # ~12 GiB. Eight ranks doing that at once landed on NUMA node 0
            # (64 GiB) and the OOM killer took the scheduler (2026-09-23).
            tensor = view[row_start:row_end]
        else:
            tensor = handle.get_tensor(name)
        with self._lock:
            self.read += 1
            if expert_id is not None and expert_id < self.n_routed:
                self.expert_ids_read.add(expert_id)
            if engram_prefix is not None:
                self.engram_tables_read.add(engram_prefix)
        return tensor

    def finish(self) -> None:
        """Call after the weight iterator is exhausted."""
        owned = self.owned_expert_ids
        missing = () if owned is None else tuple(sorted(owned - self.expert_ids_read))
        engram = self.engram_rows or {}
        missing_engram = sorted(set(engram) - self.engram_tables_read)
        logger.info(
            "DSV4 checkpoint read: kept=%d skipped=%d owned_experts=%s missing=%d mtp_only=%s narrow_engram=%s",
            self.read,
            self.skipped,
            0 if owned is None else len(owned),
            len(missing),
            self.mtp_only,
            self.engram_rows is not None,
        )
        if missing:
            preview = ", ".join(str(i) for i in missing[:8])
            raise RuntimeError(
                "DSV4 checkpoint read never yielded expert id(s) this rank "
                f"owns ({preview}). Refusing to serve with an uninitialized expert."
            )
        if missing_engram:
            raise RuntimeError(
                "DSV4 checkpoint read never yielded the Engram table(s) "
                f"{', '.join(missing_engram)}. Refusing to serve with an "
                "uninitialized table."
            )


def make_dsv4_checkpoint_reader(
    *,
    n_routed_experts: int,
    mtp_only: bool,
    engram_rows: Optional[Mapping[str, tuple[int, int]]],
    use_expert_location_metadata: bool,
) -> DSV4CheckpointReader:
    """Build the reader from the current parallel group and, for the target, EPLB metadata."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    ep_rank = int(parallel.moe_ep_rank)
    ep_size = int(parallel.moe_ep_size)
    n_routed = int(n_routed_experts)
    owned: Optional[FrozenSet[int]] = None
    if use_expert_location_metadata:
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )

        meta = get_global_expert_location_metadata()
        if meta is None:
            owned = contiguous_owned_expert_ids(n_routed, ep_rank, ep_size)
        elif int(meta.ep_size) != ep_size or int(meta.num_logical_experts) != n_routed:
            logger.warning(
                "DSV4 checkpoint read: expert-location metadata does not match "
                "this model (meta ep=%s logical=%s, model ep=%s routed=%s); "
                "not filtering expert tensors",
                getattr(meta, "ep_size", None),
                getattr(meta, "num_logical_experts", None),
                ep_size,
                n_routed,
            )
        else:
            owned = owned_expert_ids_from_physical_map(
                meta.physical_to_logical_map_cpu,
                ep_rank,
                ep_size,
                n_routed,
            )
            if owned is None:
                logger.warning(
                    "DSV4 checkpoint read: expert-location map is not a usable "
                    "EP split; not filtering expert tensors"
                )
    else:
        owned = contiguous_owned_expert_ids(n_routed, ep_rank, ep_size)
    if owned is None and ep_size > 1:
        logger.info(
            "DSV4 checkpoint read: keeping every expert tensor (ep=%d routed=%d mtp_only=%s)",
            ep_size,
            n_routed,
            mtp_only,
        )
    return DSV4CheckpointReader(
        owned_expert_ids=owned,
        n_routed=n_routed,
        mtp_only=mtp_only,
        engram_rows=engram_rows,
    )
