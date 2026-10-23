"""Experiments-as-data: named regimes, sweep files, and the CLI that consumes them.

The point of these tests is the invariant the module exists for -- that a run's
behaviour is described in exactly one place, and that the description is
recoverable from what a run writes down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nca import cli
from nca.experiments import (
    REGIMES,
    Experiment,
    experiment_from_dict,
    experiment_from_run_dir,
    load_sweep,
)
from nca.sweep import resources_from_env, sbatch_command

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = sorted((ROOT / "slurm" / "sweeps").glob("*.toml"))


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REGIMES))
def test_every_regime_resolves(name: str) -> None:
    exp = Experiment(regime=name)
    assert exp.resolved_damage_n == REGIMES[name].damage_n
    assert exp.resolved_use_pool is REGIMES[name].use_pool
    assert exp.resolved_noise_std == REGIMES[name].noise_std


def test_explicit_override_beats_the_regime() -> None:
    exp = Experiment(regime="regenerating", damage_n=0, noise_std=0.5)
    assert exp.resolved_damage_n == 0
    assert exp.resolved_noise_std == 0.5
    # An override must not leak into the regime table itself.
    assert REGIMES["regenerating"].damage_n == 3


def test_unknown_regime_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        Experiment(regime="regeneratng")  # the typo the old CLI silently ignored


def test_hidden_must_be_a_multiple_of_four() -> None:
    # The WGSL step reads whole vec4 rows, so a tail would be silently dropped.
    with pytest.raises(ValueError, match="multiple of 4"):
        Experiment(hidden=130)


def test_mismatch_p_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        Experiment(mismatch_p=3.0)


# ---------------------------------------------------------------------------
# the experiment -> TrainConfig / ModelSpec bridge
# ---------------------------------------------------------------------------


def test_cond_n_follows_the_target_count() -> None:
    assert Experiment(targets=("emoji:1f980",)).cond_n == 0
    assert Experiment(targets=("emoji:1f980", "emoji:1f419")).cond_n == 2
    assert Experiment(targets=("a", "b", "c")).spec().cond_n == 3


def test_train_config_mirrors_the_experiment() -> None:
    exp = Experiment(regime="noisy", iterations=123, unroll=(7, 9), lr=1e-3)
    cfg = exp.train_config()
    assert (cfg.iterations, cfg.lr) == (123, 1e-3)
    assert (cfg.train_steps_min, cfg.train_steps_max) == (7, 9)
    assert (cfg.use_pool, cfg.damage_n, cfg.noise_n, cfg.noise_std) == (True, 3, 4, 0.2)
    # The LR is decayed at 25% of the schedule unless told otherwise.
    assert cfg.lr_milestones == (30,)


def test_lr_milestones_resolve_late() -> None:
    # A sweep can override `iterations` without knowing the milestone.
    exp = Experiment(iterations=4000)
    assert exp.resolved_lr_milestones == (1000,)
    assert Experiment(lr_milestones=(10, 20)).train_config().lr_milestones == (10, 20)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_as_dict_round_trips() -> None:
    exp = Experiment(
        name="x.y", targets=("emoji:1f980", "emoji:1f419"), regime="noisy",
        unroll=(3, 5), mismatch_p=0.3, hidden_layers=2,
    )
    # as_dict() is resolved (regime fields become explicit), so compare behaviour
    # rather than raw dataclass equality: a second round trip must be a no-op.
    assert experiment_from_dict(exp.as_dict()).as_dict() == exp.as_dict()


def test_as_dict_is_json_safe_and_resolved() -> None:
    # `iterations` is explicit so the milestone fractions are testable, not tied to
    # whatever the default schedule happens to be.
    data = json.loads(json.dumps(Experiment(regime="noisy", iterations=8000).as_dict()))
    assert data["use_pool"] is True  # resolved, not None
    assert data["noise_n"] == 4
    assert data["lr_milestones"] == [2000]
    assert data["targets"] == ["emoji:1f980"]


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown experiment key"):
        experiment_from_dict({"regime": "growing", "damage": 3})


def test_target_shorthand() -> None:
    assert experiment_from_dict({"target": "emoji:1f980"}).targets == ("emoji:1f980",)
    with pytest.raises(ValueError, match="either"):
        experiment_from_dict({"target": "a", "targets": ["b"]})


def test_run_dir_round_trip(tmp_path: Path) -> None:
    """A run records its experiment, so the recipe survives the run directory."""
    exp = Experiment(name="crab.demo", regime="regenerating", iterations=77)
    (tmp_path / "train.json").write_text(json.dumps({"experiment_config": exp.as_dict()}))
    assert experiment_from_run_dir(tmp_path).as_dict() == exp.as_dict()


def test_legacy_run_dir_without_experiment_block(tmp_path: Path) -> None:
    """Runs trained before this refactor are rebuilt from spec.json + train.json."""
    from nca.spec import ModelSpec

    spec = ModelSpec(channel_n=8, hidden_n=32, hidden_layers=2, fire_rate=0.3)
    (tmp_path / "spec.json").write_text(json.dumps({"channel_n": spec.channel_n, "hidden_n": spec.hidden_n,
                                                     "hidden_layers": spec.hidden_layers,
                                                     "cond_n": spec.cond_n,
                                                     "fire_rate": spec.fire_rate,
                                                     "learn_perception": spec.learn_perception,
                                                     "wrap": spec.wrap}))
    (tmp_path / "train.json").write_text(
        json.dumps(
            {
                "experiment": "noisy",
                "target": "emoji:1f980",
                "target_size": 40,
                "padding": 16,
                "config": {"iterations": 2000, "damage_n": 3, "noise_n": 4, "noise_std": 0.2},
            }
        )
    )
    exp = experiment_from_run_dir(tmp_path)
    assert (exp.channels, exp.hidden, exp.hidden_layers) == (8, 32, 2)
    assert (exp.regime, exp.iterations, exp.resolved_noise_n) == ("noisy", 2000, 4)


# ---------------------------------------------------------------------------
# sweep files
# ---------------------------------------------------------------------------


def test_sweeps_exist() -> None:
    assert {p.name for p in SWEEPS} >= {"emoji-sweep.toml", "capacity.toml", "multi.toml", "cross.toml"}


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_loads(path: Path) -> None:
    sweep = load_sweep(path)
    assert sweep.runs, f"{path} has no runs"
    assert len({r.name for r in sweep.runs}) == len(sweep.runs)


def test_sweep_defaults_are_applied() -> None:
    sweep = load_sweep(ROOT / "slurm" / "sweeps" / "capacity.toml")
    assert {r.iterations for r in sweep.runs} == {8000}
    assert sweep.by_name("crab.regenerating.c=8.h=128").channels == 8
    assert sweep.by_name("crab.regenerating.c=16.h=64").hidden == 64


def test_every_exported_model_has_a_toml_recipe() -> None:
    """Every model with a sweep label must be reproducible from a committed file.

    `checkpoints` is exempt: those are extra exports of `crab.regenerating` at
    different iterations, not separate runs.
    """
    recipes = {r.name for p in SWEEPS for r in load_sweep(p).runs}
    missing = []
    for meta_path in sorted((ROOT / "exports").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("sweep") == "checkpoints":
            continue
        if meta["id"] not in recipes:
            missing.append(meta["id"])
    assert not missing, f"no sweep file trains: {missing}"


def test_duplicate_run_names_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.toml"
    path.write_text(
        "[defaults]\nregime='growing'\n\n[[run]]\nname='a'\n\n[[run]]\nname='a'\n"
    )
    with pytest.raises(ValueError, match="duplicate run name"):
        load_sweep(path)


def test_unknown_toml_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[[run]]\nname='a'\nregimee='growing'\n")
    with pytest.raises(ValueError, match="unknown experiment key"):
        load_sweep(path)


# ---------------------------------------------------------------------------
# CLI precedence: file < flags < --set
# ---------------------------------------------------------------------------


def _args(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_cli_defaults_come_from_the_experiment() -> None:
    exp = cli.experiment_for(_args())
    assert exp == Experiment()  # no drift between the CLI and the dataclass


def test_cli_flags_override_defaults() -> None:
    exp = cli.experiment_for(_args("--target", "emoji:1f980", "--experiment", "noisy",
                                   "--target-size", "32", "--hidden", "64"))
    assert (exp.targets, exp.regime, exp.target_size, exp.hidden) == (
        ("emoji:1f980",), "noisy", 32, 64)
    assert exp.resolved_noise_n == 4  # still the regime's, since no flag said otherwise


def test_config_run_and_set() -> None:
    cfg = str(ROOT / "slurm" / "sweeps" / "capacity.toml")
    exp = cli.experiment_for(_args("--config", cfg, "--run", "crab.regenerating.c=8.h=128"))
    assert (exp.channels, exp.hidden, exp.name) == (8, 128, "crab.regenerating.c=8.h=128")

    # a flag and --set both beat the file, and --set beats the flag
    exp = cli.experiment_for(_args("--config", cfg, "--run", "crab.regenerating.c=8.h=128",
                                   "--iterations", "10", "--set", "iterations=20",
                                   "--set", "targets=emoji:1f419,emoji:1f335"))
    assert exp.iterations == 20
    assert exp.targets == ("emoji:1f419", "emoji:1f335")
    assert exp.cond_n == 2


def test_config_without_run_needs_a_single_run(tmp_path: Path) -> None:
    cfg = str(ROOT / "slurm" / "sweeps" / "capacity.toml")
    with pytest.raises(SystemExit, match="--run"):
        cli.experiment_for(_args("--config", cfg))

    single = tmp_path / "one.toml"
    single.write_text("[[run]]\nname='solo'\n")
    assert cli.experiment_for(_args("--config", str(single))).name == "solo"


def test_set_unknown_field_is_an_error() -> None:
    with pytest.raises(SystemExit, match="unknown field"):
        cli.experiment_for(_args("--set", "lr_=1e-3"))


def test_regime_defaults_are_the_documented_ones() -> None:
    """The regime table is the only description of what a regime means."""
    assert {n: (r.use_pool, r.damage_n) for n, r in REGIMES.items()} == {
        "growing": (False, 0),
        "persistent": (True, 0),
        "regenerating": (True, 3),
        "noisy": (True, 3),
    }
    assert REGIMES["noisy"].noise_n == 4 and REGIMES["noisy"].noise_std == 0.2


# ---------------------------------------------------------------------------
# sbatch rendering
# ---------------------------------------------------------------------------


def test_sbatch_command_carries_placement_and_the_run() -> None:
    sweep = load_sweep(ROOT / "slurm" / "sweeps" / "cross.toml")
    run = sweep.by_name("cross8.mismatch")
    cmd = sbatch_command(sweep, run, sweep.resources, config="/abs/cross.toml")
    assert cmd[0] == "sbatch"
    assert f"--job-name={run.name}" in cmd  # squeue shows the model id
    assert "--partition=mig-priority" in cmd
    assert "--constraint=mig" in cmd
    assert cmd[-5:] == ["slurm/train.sbatch", "--config", "/abs/cross.toml", "--run", run.name]


def test_env_overrides_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = load_sweep(ROOT / "slurm" / "sweeps" / "emoji-sweep.toml")
    monkeypatch.setenv("PARTITION", "cpu")
    monkeypatch.setenv("CPUS", "2")
    res = resources_from_env(sweep.resources)
    assert (res.partition, res.cpus) == ("cpu", 2)
    assert res.constraint == "h100"  # untouched
