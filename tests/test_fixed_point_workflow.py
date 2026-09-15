"""Public API onboarding; synthetic arithmetic only."""
import torch


def test_fixed_point_primitive_public_export():
    import tikitorch.optim as optim
    from tikitorch.optim.fixed_point import fixed_point_ef_step

    assert optim.fixed_point_ef_step is fixed_point_ef_step
    assert "fixed_point_ef_step" in optim.__all__


def test_fixed_point_onboarding(capsys):
    from examples.fixed_point_workflow import main

    weights, remainder = main()
    assert weights.dtype == torch.int16 and remainder.dtype == torch.int32
    assert weights.tolist() == [9, -9]
    assert remainder.tolist() == [0, 0]
    assert "Four fractional contributions preserved" in capsys.readouterr().out
