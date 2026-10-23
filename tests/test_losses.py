"""Tests for the extra loss terms (Laplacian / SSIM / edge) and their schedule.

The contracts that matter:

* every term is zero on a perfect match (otherwise it biases the optimum),
* every term is *sensitive to where the ink is*, which is the property that made
  Gram-matrix losses the wrong tool for placed targets,
* the schedule actually suppresses the extra terms at the start of a run,
* with no ``loss`` block the objective is bit-identical to the old pixel MSE.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nca.losses import Loss, LossConfig
from nca.train import TrainConfig, loss_fn, per_sample_loss


def _rgba(size: int = 32, radius: float = 8.0, dx: float = 0.0) -> torch.Tensor:
    """A premultiplied RGBA disc: rgb where alpha is 1, transparent elsewhere.

    A *target* is 4 channels (``prepare_targets``); a *state* is ``C=16`` with the
    same first four, which is what ``_state`` builds.
    """
    coords = torch.arange(size).float() - size / 2 + 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    inside = ((xx - dx) ** 2 + yy**2) < radius**2
    x = torch.zeros(1, 4, size, size)
    x[0, 0] = inside.float()  # red
    x[0, 3] = inside.float()  # alpha
    return x


def _state(size: int = 32, radius: float = 8.0, dx: float = 0.0) -> torch.Tensor:
    """The RGBA disc as a 16-channel CA state (hidden channels zero)."""
    x = torch.zeros(1, 16, size, size)
    x[:, :4] = _rgba(size, radius, dx)
    return x


# --- config ------------------------------------------------------------------


def test_default_config_is_pixel_only_and_matches_the_old_objective():
    cfg = LossConfig.from_dict(None)
    assert not cfg.uses_perception
    assert cfg.extra_scale(0.0) == 0.0 and cfg.extra_scale(1.0) == 0.0

    x, target = _state(dx=1.0), _rgba()
    total, components = Loss(cfg)(x, target)
    assert torch.allclose(total, per_sample_loss(x, target).mean())
    assert set(components) == {"pixel"}


def test_config_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="unknown loss key"):
        LossConfig.from_dict({"laplacina": 0.5})
    with pytest.raises(ValueError, match="non-negative"):
        LossConfig(laplacian=-1.0)
    with pytest.raises(ValueError, match="fraction in"):
        LossConfig(start=2.0)
    with pytest.raises(ValueError, match="background"):
        LossConfig(background=0.5)


def test_schedule_holds_the_terms_back_then_ramps():
    cfg = LossConfig(laplacian=1.0, start=0.2, ramp=0.2)
    assert cfg.extra_scale(0.0) == 0.0
    assert cfg.extra_scale(0.19) == 0.0
    assert cfg.extra_scale(0.2) == pytest.approx(0.0)
    assert cfg.extra_scale(0.3) == pytest.approx(0.5)
    assert cfg.extra_scale(0.5) == pytest.approx(1.0)


# --- terms -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        LossConfig(laplacian=1.0),
        LossConfig(ssim=1.0),
        LossConfig(edge=1.0),
    ],
)
def test_each_term_vanishes_on_a_perfect_match(cfg):
    x, target = _state(), _rgba()
    total, components = Loss(cfg)(x, target, progress=1.0)
    assert components["pixel"] < 1e-9
    for name in ("laplacian", "ssim", "edge"):
        if name in components:
            assert components[name] < 1e-4, f"{name} is not zero at the optimum"


@pytest.mark.parametrize("term", ["laplacian", "ssim", "edge"])
def test_each_term_punishes_a_displaced_copy(term):
    """Moving the shape must cost, even though the pixel *statistics* are identical.

    This is the distinction between these terms and a Gram/style loss: the ink is
    the same amount in the same colours, only somewhere else.
    """
    cfg = LossConfig(**{term: 1.0})
    same = _rgba(dx=0.0)
    moved = _state(dx=6.0)
    _, components = Loss(cfg)(moved, same, progress=1.0)
    assert components[term] > 1e-3


def test_edge_term_is_signed_not_just_magnitude():
    """A flipped gradient must not look like a correct one."""
    cfg = LossConfig(edge=1.0)
    target = _rgba()
    inverted = _state()
    inverted[0, 0] = 1.0 - inverted[0, 0]  # same alpha, inverted red
    _, components = Loss(cfg)(inverted, target, progress=1.0)
    assert components["edge"] > 1e-3


def test_composite_is_the_renderer_rule():
    x = torch.zeros(1, 16, 4, 4)  # premultiplied: rgb=0 when alpha=0
    assert torch.allclose(Loss.composite(x, 1.0), torch.ones(1, 3, 4, 4))
    assert torch.allclose(Loss.composite(x, 0.0), torch.zeros(1, 3, 4, 4))

    x[0, 0] = 0.5  # half-transparent red over white -> (1, 0.5, 0.5)
    x[0, 3] = 0.5
    out = Loss.composite(x, 1.0)
    assert out[0, 0].max() == pytest.approx(1.0)
    assert out[0, 1].max() == pytest.approx(0.5)


def test_terms_are_scale_free_across_targets():
    """A weight must mean the same thing for a bold and a faint target."""
    cfg = LossConfig(edge=1.0, laplacian=1.0)
    # Same silhouette and the same displacement; only the contrast differs.
    _, bold = Loss(cfg)(_state(dx=4.0), _rgba(), progress=1.0)

    faint_target = _rgba()
    faint_target[:, :3] *= 0.2
    faint_state = _state(dx=4.0)
    faint_state[:, :3] *= 0.2
    _, faint = Loss(cfg)(faint_state, faint_target, progress=1.0)

    for term in ("edge", "laplacian"):
        assert 0.5 < faint[term] / bold[term] < 2.0, f"{term} is not contrast-normalised"


def test_calibration_makes_a_weight_mean_a_fraction_of_the_pixel_loss():
    """Uncalibrated, the raw terms are 20-40x the pixel error and swamp it.

    On the damaged heart used by `scripts/loss_ab.py`: pixel 0.024, laplacian 1.0,
    edge 0.96. Weighting those directly is not a "0.5 term", it is a term 20x
    heavier than the objective it is supposed to complement -- and it wins, and the
    pattern dissolves.
    """
    cfg = LossConfig(laplacian=0.5, edge=0.05)
    target = _rgba()
    damaged = _state()
    damaged[..., 8:16, 8:16] = 0.0

    raw = Loss(LossConfig(laplacian=0.5, edge=0.05, normalize=False))
    calibrated = Loss(cfg)
    scales = calibrated.calibrate(damaged, target)
    assert scales["laplacian"] < 0.1 and scales["edge"] < 0.1

    total_raw, raw_parts = raw(damaged, target)
    total_cal, cal_parts = calibrated(damaged, target)
    # Same terms; only their scale changed.
    assert torch.allclose(raw_parts["laplacian"], cal_parts["laplacian"])
    # Calibrated, the extras start at the same order of magnitude as the pixel term.
    assert total_cal < 3.0 * cal_parts["pixel"]
    assert total_raw > 10.0 * total_cal


def test_calibration_is_a_no_op_without_extra_terms():
    criterion = Loss(LossConfig())
    assert criterion.calibrate(_state(), _rgba()) == {}
    assert criterion.calibrated


# --- wiring ------------------------------------------------------------------


def test_train_config_records_the_loss_block(tmp_path):
    from dataclasses import asdict

    cfg = TrainConfig(loss=LossConfig(laplacian=0.5).as_dict())
    assert asdict(cfg)["loss"]["laplacian"] == 0.5
    # ...and the default records nothing, so old runs stay comparable.
    assert asdict(TrainConfig())["loss"] is None


def test_experiment_validates_the_loss_table():
    from nca.experiments import Experiment

    exp = Experiment(loss={"laplacian": 0.5, "ssim": 0.1, "start": 0.2})
    assert exp.train_config().loss is not None
    with pytest.raises(ValueError, match="unknown loss key"):
        Experiment(loss={"laphacian": 0.5})


def test_pixel_only_loss_fn_is_untouched():
    x, target = _state(dx=2.0), _rgba()
    assert torch.allclose(loss_fn(x, target), per_sample_loss(x, target).mean())
