"""Experiments as data: one ``Experiment`` dataclass, named regimes, sweep files.

Before this module every training default lived in two places -- an argparse
``default=`` and a per-regime ``EXPERIMENTS[...]`` dict that the CLI then merged
with ``preset[key] if args.x is None else args.x``. Now a default lives exactly
once, on a field of :class:`Experiment`, and three callers merely *supply* one:

* the CLI, for an ad-hoc run (``main.py --target ... --experiment ...``),
* a sweep file (``slurm/sweeps/*.toml``), for a study,
* :func:`experiment_from_run_dir`, for reading a run back.

A run records its resolved experiment in ``train.json["experiment"]``; the
exporter copies it into ``meta.json["train"]``, so an exported model is
reproducible from ``exports/`` alone -- no shell history, no run directory.

    >>> from nca.experiments import Experiment, load_sweep
    >>> e = Experiment(name="demo", targets=("emoji:1f980",), regime="noisy")
    >>> e.use_pool, e.noise_std          # regime defaults, resolved
    (True, 0.2)
    >>> e.train_config().iterations      # the quick local default; studies set their own
    1200
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .spec import ModelSpec
from .train import TrainConfig

# ---------------------------------------------------------------------------
# Regimes: the only place "what does `regenerating` mean" is written down.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """How a model is asked to behave during training.

    ``use_pool`` keeps a pool of previously-seen states (the paper's trick for
    stable long-run behaviour), ``damage_n`` punches holes in that many pool
    samples per batch, and the ``noise_*`` pair corrupts pool samples with
    Gaussian state noise instead of / as well as occluding them.
    """

    name: str
    use_pool: bool
    damage_n: int
    noise_n: int = 0
    noise_std: float = 0.0
    help: str = ""


REGIMES: dict[str, Regime] = {
    r.name: r
    for r in (
        Regime(
            "growing",
            use_pool=False,
            damage_n=0,
            help="grow from the seed only: fast, but gets stuck in a fixed point",
        ),
        Regime(
            "persistent",
            use_pool=True,
            damage_n=0,
            help="keep a sample pool: stable patterns that stay put",
        ),
        Regime(
            "regenerating",
            use_pool=True,
            damage_n=3,
            help="pool + holes punched in 3 samples per batch: self-healing",
        ),
        Regime(
            "noisy",
            use_pool=True,
            damage_n=3,
            noise_n=4,
            noise_std=0.2,
            help="regenerating + Gaussian state noise: rebuild from a corrupted state",
        ),
    )
}

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """One training run: what to grow, with what, and under which regime.

    Fields whose value depends on the regime (``use_pool``, ``damage_n``,
    ``noise_n``, ``noise_std``) stay ``None`` here and are resolved through the
    properties below, so a regime is described once (in :data:`REGIMES`) and an
    explicit override still wins.
    """

    name: str = ""
    # One target = an unconditional model; two or more = a conditional one whose
    # cond_n-way one-hot selects the target.
    targets: tuple[str, ...] = ("emoji:1f980",)
    regime: str = "regenerating"

    # Canvas: the grid the CA runs on is target_size + 2*padding. These are the
    # *quick local* defaults (a laptop run in ~10 min); every study in
    # slurm/sweeps/ states its own in [defaults].
    target_size: int = 32
    padding: int = 8

    # Shape of the model (the `ModelSpec` subset a training run chooses).
    channels: int = 16
    hidden: int = 128
    hidden_layers: int = 1
    fire_rate: float = 0.5
    learn_perception: bool = False
    wrap: bool = True

    # Regime overrides. None = take it from the regime.
    use_pool: bool | None = None
    damage_n: int | None = None
    noise_n: int | None = None
    noise_std: float | None = None

    # Fraction of each batch asked to convert a grown target into another one.
    # Only meaningful with several targets.
    mismatch_p: float = 0.0

    # Training schedule.
    pool_size: int = 1024
    batch_size: int = 4
    iterations: int = 1200
    unroll: tuple[int, int] = (48, 64)  # CA steps unrolled per training step
    lr: float = 2e-3
    lr_milestones: tuple[int, ...] | None = None  # None = 25% of iterations
    lr_gamma: float = 0.1
    normalize_grads: bool = True
    amp: str = "off"
    seed: int = 0
    log_every: int = 10
    ckpt_every: int = 100

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}; known: {sorted(REGIMES)}")
        if not self.targets:
            raise ValueError("an experiment needs at least one target")
        if self.hidden % 4:
            raise ValueError(
                f"hidden {self.hidden} is not a multiple of 4 (the WGSL step reads "
                f"whole vec4 rows, so the tail would be silently dropped)"
            )
        if self.unroll[0] > self.unroll[1]:
            raise ValueError(f"unroll={self.unroll}: min > max")
        if not 0.0 <= self.mismatch_p <= 1.0:
            raise ValueError(f"mismatch_p={self.mismatch_p} is not a probability")

    # -- resolved regime ----------------------------------------------------

    @property
    def regime_spec(self) -> Regime:
        return REGIMES[self.regime]

    @property
    def resolved_use_pool(self) -> bool:
        return self.regime_spec.use_pool if self.use_pool is None else self.use_pool

    @property
    def resolved_damage_n(self) -> int:
        return self.regime_spec.damage_n if self.damage_n is None else self.damage_n

    @property
    def resolved_noise_n(self) -> int:
        return self.regime_spec.noise_n if self.noise_n is None else self.noise_n

    @property
    def resolved_noise_std(self) -> float:
        return self.regime_spec.noise_std if self.noise_std is None else self.noise_std

    @property
    def resolved_lr_milestones(self) -> tuple[int, ...]:
        if self.lr_milestones is not None:
            return tuple(self.lr_milestones)
        return (max(1, int(0.25 * self.iterations)),)

    @property
    def cond_n(self) -> int:
        """0 for a single-target model; the one-hot width for a conditional one."""
        return len(self.targets) if len(self.targets) > 1 else 0

    @property
    def grid(self) -> int:
        return self.target_size + 2 * self.padding

    # -- consumers ---------------------------------------------------------

    def train_config(self) -> TrainConfig:
        """The training-loop half of the experiment (``TrainConfig``)."""
        return TrainConfig(
            iterations=self.iterations,
            batch_size=self.batch_size,
            pool_size=self.pool_size,
            use_pool=self.resolved_use_pool,
            damage_n=self.resolved_damage_n,
            noise_n=self.resolved_noise_n,
            noise_std=self.resolved_noise_std,
            mismatch_p=self.mismatch_p,
            train_steps_min=self.unroll[0],
            train_steps_max=self.unroll[1],
            lr=self.lr,
            lr_milestones=self.resolved_lr_milestones,
            lr_gamma=self.lr_gamma,
            normalize_grads=self.normalize_grads,
            log_every=self.log_every,
            ckpt_every=self.ckpt_every,
            save_pool=False,  # run-control, not part of the experiment
            make_gif=True,
            gif_steps=120,
            seed=self.seed,
            amp=self.amp,
        )

    def spec(self) -> ModelSpec:
        """The model shape half of the experiment."""
        return ModelSpec(
            channel_n=self.channels,
            hidden_n=self.hidden,
            hidden_layers=self.hidden_layers,
            cond_n=self.cond_n,
            fire_rate=self.fire_rate,
            learn_perception=self.learn_perception,
            wrap=self.wrap,
        )

    def as_dict(self) -> dict[str, Any]:
        """Fully resolved, JSON-safe form -- this is what a run records.

        Resolved (not ``None``) so the dict alone can rebuild the experiment: that
        is the point of putting it in ``meta.json``.
        """
        out = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in ("use_pool", "damage_n", "noise_n", "noise_std", "lr_milestones")
        }
        out["use_pool"] = self.resolved_use_pool
        out["damage_n"] = self.resolved_damage_n
        out["noise_n"] = self.resolved_noise_n
        out["noise_std"] = self.resolved_noise_std
        out["lr_milestones"] = list(self.resolved_lr_milestones)
        out["targets"] = list(self.targets)
        out["unroll"] = list(self.unroll)
        return out

    def with_overrides(self, **kwargs: Any) -> Experiment:
        """``experiment`` with the non-``None`` overrides applied."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


ALIASES = {"target": "targets", "channels": "channels", "hidden": "hidden"}


def experiment_from_dict(data: dict[str, Any]) -> Experiment:
    """Build an :class:`Experiment` from a TOML/JSON mapping, strictly.

    ``target`` (singular) is accepted as a spelling of ``targets``; anything else
    unknown is an error, because a silently ignored key is a run that quietly
    trains something other than what its file says.
    """
    data = dict(data)
    if "target" in data:
        single = data.pop("target")
        if "targets" in data:
            raise ValueError("give either `target` or `targets`, not both")
        data["targets"] = [single]
    for key in ("targets", "unroll", "lr_milestones"):
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])
    known = {f.name for f in fields(Experiment)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown experiment key(s): {unknown}; known: {sorted(known)}")
    return Experiment(**data)


def experiment_from_run_dir(run_dir: str | Path) -> Experiment:
    """Read a run's experiment back out of its ``train.json``.

    Runs written before Phase 3 have no ``experiment`` block; they are rebuilt
    from ``spec.json`` + ``train.json["config"]`` so old runs stay loadable.
    """
    import json

    run_dir = Path(run_dir)
    train = json.loads((run_dir / "train.json").read_text())
    if "experiment_config" in train:
        return experiment_from_dict(train["experiment_config"])
    spec = ModelSpec.from_run_dir(run_dir)
    cfg = train["config"]
    return Experiment(
        name=run_dir.name,
        targets=tuple(train.get("targets") or [train.get("target") or "unknown"]),
        regime=train.get("experiment", cfg.get("regime", "regenerating")),
        target_size=int(train.get("target_size") or 40),
        padding=int(train.get("padding") or 16),
        channels=spec.channel_n,
        hidden=spec.hidden_n,
        hidden_layers=spec.hidden_layers,
        fire_rate=spec.fire_rate,
        learn_perception=spec.learn_perception,
        wrap=spec.wrap,
        use_pool=cfg.get("use_pool"),
        damage_n=cfg.get("damage_n"),
        noise_n=cfg.get("noise_n"),
        noise_std=cfg.get("noise_std"),
        mismatch_p=cfg.get("mismatch_p", 0.0),
        pool_size=cfg.get("pool_size", 1024),
        batch_size=cfg.get("batch_size", 16),
        iterations=cfg.get("iterations", 8000),
        unroll=(cfg.get("train_steps_min", 64), cfg.get("train_steps_max", 96)),
        lr=cfg.get("lr", 2e-3),
        lr_milestones=tuple(cfg.get("lr_milestones") or ()) or None,
        lr_gamma=cfg.get("lr_gamma", 0.1),
        normalize_grads=cfg.get("normalize_grads", True),
        amp=cfg.get("amp", "off"),
        seed=cfg.get("seed", 0),
        log_every=cfg.get("log_every", 25),
        ckpt_every=cfg.get("ckpt_every", 250),
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resources:
    """Slurm placement for a sweep (overridable from the environment)."""

    partition: str = "gpu"
    constraint: str = "h100"
    gres: str = "gpu:1"
    time: str = "12:00:00"
    mem: str = "64G"
    cpus: int = 8


@dataclass(frozen=True)
class Sweep:
    name: str
    group: str
    resources: Resources
    runs: tuple[Experiment, ...] = field(default_factory=tuple)

    def by_name(self, name: str) -> Experiment:
        for run in self.runs:
            if run.name == name:
                return run
        raise KeyError(f"{self.name}: no run named {name!r}; have {[r.name for r in self.runs]}")


def load_sweep(path: str | Path) -> Sweep:
    """Parse ``slurm/sweeps/*.toml`` into a :class:`Sweep`.

    ``[defaults]`` is merged into every ``[[run]]`` so a study states its shared
    knobs once; ``[resources]`` is the sbatch placement.
    """
    path = Path(path)
    data = tomllib.loads(path.read_text())
    if "run" not in data:
        raise ValueError(f"{path}: no [[run]] entries")
    defaults = data.get("defaults", {})
    unknown_top = sorted(set(data) - {"sweep", "defaults", "resources", "run"})
    if unknown_top:
        raise ValueError(f"{path}: unknown section(s) {unknown_top}")
    meta = data.get("sweep", {})
    resources = Resources(**data.get("resources", {}))
    runs: list[Experiment] = []
    for entry in data["run"]:
        merged = {**defaults, **entry}
        merged.setdefault("name", entry.get("name") or f"{path.stem}-{len(runs)}")
        runs.append(experiment_from_dict(merged))
    names = [r.name for r in runs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate run name(s) {dupes}")
    return Sweep(name=meta.get("name", path.stem), group=meta.get("group", path.stem), resources=resources, runs=tuple(runs))
