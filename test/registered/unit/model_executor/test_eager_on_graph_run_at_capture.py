"""``eager_on_graph(run_at_capture=True)``: the break contract the SM70
DeepSeek-V4.1 breaks use (run at capture, return a value, copy back on replay)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def recording_capture():
    events = []
    graph = bcg.BreakableCUDAGraph()
    capture = SimpleNamespace(
        cuda_graph=graph,
        _end_current_segment=lambda: events.append("end"),
        _begin_new_segment=lambda: events.append("begin"),
    )
    token = bcg._current_capture_var.set(capture)
    try:
        yield graph, events
    finally:
        bcg._current_capture_var.reset(token)


def test_outside_capture_returns_the_value():
    double = bcg.eager_on_graph(lambda x: x * 2, run_at_capture=True)
    assert double(torch.tensor([3.0])).item() == 6


def test_legacy_bool_form_takes_run_at_capture():
    double = bcg.eager_on_graph(True, run_at_capture=True)(lambda x: x * 2)
    with recording_capture() as (graph, events):
        assert double(torch.tensor([3.0])).item() == 6
    assert events == ["end", "begin"]
    assert len(graph._break_fns) == 1


def test_runs_at_capture_and_replay_copies_into_the_captured_output():
    calls = []

    @bcg.eager_on_graph(run_at_capture=True)
    def double(x):
        calls.append(x.item())
        return x * 2

    x = torch.tensor([3.0])
    with recording_capture() as (graph, events):
        captured = double(x)
    assert events == ["end", "begin"]
    assert calls == [3.0]
    assert captured.item() == 6

    x.fill_(5.0)
    live = graph._break_fns[0]()
    assert calls == [3.0, 5.0]
    assert live.item() == 10
    # The next captured segment reads the capture-time address.
    assert captured.item() == 10
    assert live.data_ptr() != captured.data_ptr()


def test_tuple_output_is_copied_per_element():
    @bcg.eager_on_graph(run_at_capture=True)
    def split(x):
        return x + 1, x - 1

    x = torch.tensor([1.0])
    with recording_capture() as (graph, _events):
        plus, minus = split(x)
    x.fill_(4.0)
    graph._break_fns[0]()
    assert (plus.item(), minus.item()) == (5.0, 3.0)


def test_nested_break_runs_inside_the_outer_body():
    @bcg.eager_on_graph(run_at_capture=True)
    def inner(x):
        return x + 1

    @bcg.eager_on_graph
    def inner_none(out):
        out.add_(10)

    @bcg.eager_on_graph(run_at_capture=True)
    def outer(x):
        y = inner(x)
        inner_none(y)
        return y * 2

    x = torch.tensor([1.0])
    with recording_capture() as (graph, events):
        assert outer(x).item() == 24
    # One break for the outer body; the nested ones ran in it.
    assert events == ["end", "begin"]
    assert len(graph._break_fns) == 1
    assert bcg._run_at_capture_body_var.get() is False

    x.fill_(2.0)
    assert graph._break_fns[0]().item() == 26
