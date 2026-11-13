"""Admission counts a session pin's row once: free while idle, held once lent."""

import dataclasses
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.pool_stats_observer import (  # noqa: E402
    SchedulerPoolStatsObserver,
)
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NUM_ROWS = 4


def _scheduler(pin, running, free_rows):
    fields = {f.name: None for f in dataclasses.fields(SchedulerPoolStatsObserver)}
    fields["tree_cache"] = SimpleNamespace(session_records=lambda: {"pin": pin})
    observer = SchedulerPoolStatsObserver(**fields)
    s = Scheduler.__new__(Scheduler)
    s.pool_stats_observer = observer
    s.req_to_token_pool = SimpleNamespace(available_size=lambda: free_rows)
    s.beam_coordinator = SimpleNamespace(
        pending_member_rows=lambda batch: 0, pending_member_rows_of=lambda reqs: 0
    )
    s.running_batch = ScheduleBatch(reqs=running)
    return s


def _allocatable(s, running_bs, counted):
    with get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS):
        return s.get_num_allocatable_reqs(
            running_bs,
            num_held_rows=sum(r.kv.holds_kv for r in counted),
            admitted_reqs=counted,
        )


class TestIdleSessionRows(CustomTestCase):
    def setUp(self):
        self.pin = SimpleNamespace(holds_kv=True)

    def test_idle_pin_counts_as_allocatable(self):
        s = _scheduler(self.pin, running=[], free_rows=0)
        self.assertEqual(_allocatable(s, 0, counted=[]), 1)

    def test_pin_lent_to_counted_request_counts_once(self):
        s = _scheduler(self.pin, running=[], free_rows=0)
        borrower = SimpleNamespace(kv=self.pin)
        self.assertEqual(_allocatable(s, 0, counted=[borrower]), 1)

    def test_pin_lent_to_running_request_is_not_free(self):
        running = [SimpleNamespace(kv=self.pin)]
        s = _scheduler(self.pin, running=running, free_rows=0)
        self.assertEqual(_allocatable(s, len(running), counted=[]), 0)


if __name__ == "__main__":
    unittest.main()
