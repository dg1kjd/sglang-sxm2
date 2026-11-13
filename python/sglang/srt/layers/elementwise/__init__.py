"""Volta (sm70) element-wise kernels.

The generic element-wise kernels that used to live in
``sglang/srt/layers/elementwise.py`` moved upstream to
``sglang.kernels.ops.elementwise.elementwise`` (RFC #29630).  They are
re-exported here so any remaining ``sglang.srt.layers.elementwise`` import keeps
resolving; the canonical implementations are upstream's.

The module beside this one, ``sigmoid_mul``, is Volta-specific and has no
upstream equivalent, which is why this package exists at all. (Its old
``fast_topk`` loader duplicated ``sglang.kernels.ops.attention.fast_topk`` and
was removed once QSA moved to ``ordered_topk``.) (``hc_combine`` and ``hc_mix`` were copies of upstream's
``sglang.kernels.ops.elementwise`` modules and were removed.)

Note: base's module also defined ``fused_softcap*``, ``experts_combine*``,
``Softcap`` and ``FusedDualResidualRMSNorm``.  Upstream deleted all of them and
nothing in this tree calls them, so they are not carried forward.
"""

from sglang.kernels.ops.elementwise.elementwise import *  # noqa: F401,F403
