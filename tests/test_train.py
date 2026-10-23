"""Smoke tests for the training loop.

These are deliberately tiny (a 16x16 grid, 2 iterations, CPU) -- they are not
about loss quality but about the loop *running* and about a run directory being
self-describing afterwards. The single-target / batch>1 case is a regression
test: ``tgt`` used to stay ``[1, C, H, W]`` on non-conditional runs and then be
indexed per sample, which raised ``IndexError: index 1 is out of bounds for
dimension 0 with size 1`` for every batch_size > 1.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from nca.spec import ModelSpec  # noqa: E402
from nca.targets import make_seed  # noqa: E402
from nca.train import TrainConfig, train  # noqa: E402

GRID = 16
CHANNELS = 8


def _cfg(**kwargs) -> TrainConfig:
    base = dict(
        iterations=2,
        batch_size=4,
        pool_size=16,
        train_steps_min=2,
        train_steps_max=3,
        log_every=100,
        progress_every=0,
        make_gif=False,
        ckpt_every=10**9,
    )
    base.update(kwargs)
    return TrainConfig(**base)


def _target(n_targets: int = 1, cond_n: int = 0) -> torch.Tensor:
    # Targets are always 4 channels (RGB + alpha): the loss only looks at those.
    tgt = torch.zeros(n_targets, 4, GRID, GRID)
    for i in range(n_targets):
        tgt[i, :3] = torch.rand(3, GRID, GRID) * 0.5 + 0.5 * (i + 1) / n_targets
    tgt[:, 3] = 1.0
    if cond_n:
        tgt = tgt.expand(cond_n, *tgt.shape[1:])
    return tgt


def _model(cond_n: int = 0):
    from nca.model import NeuralCA

    return NeuralCA(channel_n=CHANNELS, hidden_n=8, cond_n=cond_n)


def test_single_target_training_runs_and_describes_the_run(tmp_path):
    ca = _model()
    losses = train(
        ca,
        _target(1),
        make_seed(GRID, GRID, CHANNELS),
        _cfg(),
        torch.device("cpu"),
        tmp_path,
        provenance={"experiment": "regenerating", "target": "shape:heart"},
    )

    assert len(losses) == 2
    assert all(loss == loss and loss < float("inf") for loss in losses)  # not NaN/inf
    assert (tmp_path / "ckpt_final.pt").exists()
    assert (tmp_path / "loss.json").exists()

    # spec.json + train.json make the run directory loadable without the CLI flags.
    spec = ModelSpec.from_run_dir(tmp_path)
    assert spec == ca.spec
    train_json = json.loads((tmp_path / "train.json").read_text())
    assert train_json["experiment"] == "regenerating"
    assert train_json["spec"]["channel_n"] == CHANNELS
    assert train_json["config"]["iterations"] == 2


def test_conditional_training_with_mismatch_runs(tmp_path):
    ca = _model(cond_n=2)
    losses = train(
        ca,
        _target(2, cond_n=2),
        make_seed(GRID, GRID, CHANNELS),
        _cfg(mismatch_p=0.5),
        torch.device("cpu"),
        tmp_path,
    )
    assert len(losses) == 2
    assert "cond=2" in str(ca.spec)


def test_resume_continues_the_loss_history(tmp_path):
    ca = _model()
    train(ca, _target(1), make_seed(GRID, GRID, CHANNELS), _cfg(), torch.device("cpu"), tmp_path)
    assert len(json.loads((tmp_path / "loss.json").read_text())) == 2

    ca2 = _model()
    losses = train(
        ca2,
        _target(1),
        make_seed(GRID, GRID, CHANNELS),
        _cfg(iterations=4),
        torch.device("cpu"),
        tmp_path,
        resume=tmp_path / "state.pt",
    )
    assert len(losses) == 4
