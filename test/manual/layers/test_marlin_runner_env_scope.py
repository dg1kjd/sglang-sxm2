def test_fused_experts_none_to_marlin_reads_module_envs():
    # A function-local `from sglang.srt.environ import envs` anywhere in the
    # body makes every earlier `envs` read in it an UnboundLocalError.
    from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin

    assert "envs" not in fused_experts_none_to_marlin.__code__.co_varnames
