"""Training loop for the Neural CA.

Implements the "sample pool" + damage strategy from the paper, plus the
per-variable gradient normalisation that stabilises training.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import NeuralCA
from .pool import SamplePool
from .viz import growth_frames, plot_loss, render, save_gif, save_image, tile


@dataclass
class TrainConfig:
    iterations: int = 2000
    batch_size: int = 8
    pool_size: int = 1024
    use_pool: bool = True
    damage_n: int = 3
    noise_n: int = 0
    noise_std: float = 0.0
    # Fraction of the batch whose *conditioning* is reassigned to a different
    # target while keeping its state, i.e. "turn this emoji into another one".
    mismatch_p: float = 0.0
    train_steps_min: int = 64
    train_steps_max: int = 96
    lr: float = 2e-3
    lr_milestones: tuple[int, ...] = (2000,)
    lr_gamma: float = 0.1
    normalize_grads: bool = True
    log_every: int = 10
    ckpt_every: int = 100
    progress_every: int = 100  # render a batch montage every N iterations
    save_pool: bool = False  # include the (large) sample pool in state.pt
    gif_steps: int = 120
    gif_frame_every: int = 2
    gif_scale: int = 3
    make_gif: bool = True
    seed: int = 0
    amp: str = "off"  # off | bf16


def per_sample_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the RGBA channels, reduced to ``[B]``."""
    diff = x[:, :4] - target
    return (diff * diff).mean(dim=(1, 2, 3))


def loss_fn(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return per_sample_loss(x, target).mean()


def make_circle_masks(
    n: int, h: int, w: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Random circular damage masks, shape ``[n, h, w]`` (1 = keep)."""
    x = torch.linspace(-1.0, 1.0, w)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h)[None, :, None]
    center = torch.rand(2, n, 1, 1, generator=generator) - 0.5
    r = torch.rand(n, 1, 1, generator=generator) * 0.3 + 0.1
    x = (x - center[0]) / r
    y = (y - center[1]) / r
    return (x * x + y * y < 1.0).float()


def _train_step(
    ca: NeuralCA,
    forward,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    target: torch.Tensor,
    cfg: TrainConfig,
    masks: torch.Tensor,
    cond: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One optimiser step over the unroll. Returns ``(final_state, loss)``.

    ``loss`` is detached but **left on the device**: the caller materialises the
    number only when it actually needs it, so we don't force a sync every step.
    """
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp == "bf16"
    ):
        x = forward(x0, masks, cond)
    loss = loss_fn(x, target)  # fp32, so the reported loss stays comparable

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.normalize_grads:
        grads = [p.grad for p in ca.parameters() if p.grad is not None]
        if grads:
            norms = torch._foreach_norm(grads)
            torch._foreach_div_(grads, [n + 1e-8 for n in norms])
    optimizer.step()
    return x.detach(), loss.detach()


def _save_resume_state(
    path: Path,
    ca: NeuralCA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    gpu_generator: torch.Generator,
    device: torch.device,
    step: int,
    losses: list[float],
    pool: SamplePool | None,
    save_pool: bool = False,
) -> None:
    state = {
        "model": ca.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "losses": losses,
        "rng": {
            "cpu": torch.get_rng_state(),
            "gen": generator.get_state(),
            "gen_gpu": gpu_generator.get_state(),
        },
    }
    if device.type == "cuda":
        state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    # The pool is [pool_size, C, H, W] (~340 MB at 1024x16x72x72). It is the whole
    # point of state.pt being large, and resuming without it is fine: the pool
    # simply restarts from the seed. Only keep it on request.
    if save_pool and pool is not None:
        state["pool"] = pool.x
    torch.save(state, path)


def train(
    ca: NeuralCA,
    target: torch.Tensor,
    seed: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    resume: str | Path | None = None,
    provenance: dict | None = None,
) -> list[float]:
    """Train ``ca`` in place and return the loss history.

    ``provenance`` is free-form metadata recorded next to the run (``train.json``)
    -- the CLI passes the target spec(s) / experiment name so a run directory is
    self-describing without having to read the shell history.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A resume has to be validated *before* the run directory is rewritten: if the
    # checkpoint does not fit the module we just built, `load_state_dict` raises,
    # and a spec.json written earlier would then describe a model that no
    # checkpoint in the directory matches (`ModelSpec.from_run_dir` refuses that,
    # but not writing the lie in the first place is better).
    resume_state: dict | None = None
    if resume is not None:
        resume_state = torch.load(resume, map_location="cpu")
        if "model" not in resume_state:
            # `ckpt_*.pt` is a bare state dict; only `state.pt` carries the optimizer
            # and RNG, which is what resuming needs.
            raise ValueError(
                f"{resume} is a bare checkpoint (no optimizer/RNG state); "
                f"--resume needs the run's state.pt"
            )
        ca.load_state_dict(resume_state["model"])

    # Record what this run *is* before any training happens, so a crashed or
    # still-running job is already loadable by `nca.sample --run <dir>` and
    # re-exportable without the caller remembering the flags.
    spec = ca.spec
    spec.save(out_dir / "spec.json")
    (out_dir / "train.json").write_text(
        json.dumps(
            {
                **(provenance or {}),
                "config": asdict(cfg),
                "spec": asdict(spec),
                "device": str(device),
                "torch_version": torch.__version__,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n"
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    # Separate generator for the on-device fire masks (drawn a whole unroll at a
    # time). Falls back to CPU behaviour on a CPU device.
    gpu_generator = torch.Generator(device=device.type)
    gpu_generator.manual_seed(cfg.seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    ca.to(device)

    # The unroll is ~80 sequential steps but it is throughput-bound, not
    # latency-bound: time is linear in batch size, so there is nothing to fuse
    # away. `torch.compile` measured neutral-to-slower here (compiling the whole
    # sequence pays a warmup and is specialised per unroll length) - see
    # docs/findings/training-perf.md. `scripts/profile_train.py --cuda-graphs`
    # re-measures it if the assumption ever needs revisiting.
    def forward(
        x: torch.Tensor, masks: torch.Tensor, cond: torch.Tensor | None
    ) -> torch.Tensor:
        for k in range(masks.shape[0]):
            x = ca(x, update_mask=masks[k], cond=cond)
        return x
    target = target.to(device)
    seed = seed.to(device)
    _, h, w = seed.shape

    # A conditional run gets one target per emoji; every cell is told which one to
    # grow via a one-hot vector appended to its perception.
    n_targets = int(target.shape[0])
    cond_n = n_targets if n_targets > 1 else 0
    if ca.cond_n != cond_n:
        raise ValueError(
            f"model built with cond_n={ca.cond_n} but {n_targets} target(s) given"
        )

    optimizer = torch.optim.Adam(ca.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )

    pool = None
    if cfg.use_pool:
        pool_x = seed.unsqueeze(0).repeat(cfg.pool_size, 1, 1, 1).contiguous()
        pool_cond = None
        if cond_n:
            # Spread the emoji round-robin over the initial pool.
            pool_cond = F.one_hot(torch.arange(cfg.pool_size) % cond_n, cond_n).float().to(device)
        pool = SamplePool(pool_x, pool_cond)

    # Keep at least one slot (the seed) untouched by damage.
    damage_n = min(cfg.damage_n, max(cfg.batch_size - 1, 0))
    if damage_n != cfg.damage_n:
        print(f"note: clamping damage_n {cfg.damage_n} -> {damage_n} (batch_size={cfg.batch_size})")
    # Noise goes on the slots just before the damaged ones (the seed stays clean).
    noise_n = min(cfg.noise_n, max(cfg.batch_size - 1 - damage_n, 0))
    if noise_n != cfg.noise_n:
        print(f"note: clamping noise_n {cfg.noise_n} -> {noise_n} (batch_size={cfg.batch_size})")
    if noise_n and cfg.noise_std <= 0:
        print("note: noise_n > 0 but noise_std is 0; no noise will be applied")

    losses: list[float] = []
    start_step = 0
    if resume_state is not None:
        state = resume_state
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state.get("step", 0))
        losses = list(state.get("losses", []))
        rng = state.get("rng", {})
        if "cpu" in rng:
            torch.set_rng_state(rng["cpu"])
        if "gen" in rng:
            generator.set_state(rng["gen"])
        if "gen_gpu" in rng:
            gpu_generator.set_state(rng["gen_gpu"])
        if device.type == "cuda" and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if pool is not None:
            if "pool" in state:
                pool.x = state["pool"].to(device)
            else:
                print("note: no pool in resume state; restarting the pool from the seed")
        print(f"resumed from {resume} at step {start_step}")

    start = time.time()

    print(
        f"Training on {device} | grid {h}x{w} | {ca.num_parameters} params | "
        f"pool={cfg.use_pool} damage={damage_n} noise={noise_n}x{cfg.noise_std} "
        f"targets={n_targets} mismatch={cfg.mismatch_p} "
        f"| {cfg.iterations} iters"
    )

    # Loss values stay as device scalars and are only turned into floats when a
    # log line / checkpoint actually needs them (no per-step sync).
    pending: list[torch.Tensor] = []

    def flush_losses() -> None:
        if pending:
            losses.extend(torch.stack(pending).tolist())
            pending.clear()

    for step in range(start_step, cfg.iterations):
        tgt = target
        if pool is not None:
            idx, x0, cond = pool.sample(cfg.batch_size, generator)
            if cond is not None:
                tgt = target[cond.argmax(dim=1)]
            else:
                # Non-conditional run: every slot wants the same target. Expand it
                # to a per-sample tensor so the ranking / mismatch code below can
                # index it (tgt[rank]) -- relying on broadcast silently breaks the
                # moment batch_size > 1.
                tgt = target.expand(cfg.batch_size, *target.shape[1:])
            with torch.no_grad():
                rank = per_sample_loss(x0, tgt).argsort(descending=True)
            x0 = x0[rank]
            idx = idx[rank]
            tgt = tgt[rank]
            if cond is not None:
                cond = cond[rank]
            # Always include the original seed so the model never forgets how
            # to grow from scratch. Rotate which emoji it grows into so every
            # target gets exercised from the seed.
            x0[0] = seed
            if cond is not None:
                sel = step % cond_n
                cond[0] = F.one_hot(torch.tensor(sel, device=device), cond_n).float()
                tgt[0] = target[sel]
            if cond is not None and cfg.mismatch_p > 0:
                # Break the state/cond pairing for a slice of the batch (never the
                # seed slot): an already-grown emoji is asked to become a
                # *different* one. Without this every example is self-consistent,
                # so a small MLP could satisfy the task by memorising N attractors
                # and treating the one-hot as an index. Cross-target pairs force the
                # conditioning to actually steer the dynamics - and a model that can
                # convert cannot have collapsed its condition.
                k = min(int(round(cfg.mismatch_p * cfg.batch_size)), cfg.batch_size - 1)
                if k > 0:
                    # indices 1..batch-1, so the fresh seed at 0 is untouched
                    cand = (torch.randperm(cfg.batch_size - 1, generator=generator)[:k] + 1).to(device)
                    cur = cond[cand].argmax(dim=1).cpu()
                    # any target other than the current one
                    delta = torch.randint(0, cond_n - 1, (k,), generator=generator)
                    newt = ((cur + 1 + delta) % cond_n).to(device)
                    cond[cand] = F.one_hot(newt, cond_n).float()
                    tgt[cand] = target[newt]
            if damage_n:
                dmg = make_circle_masks(damage_n, h, w, generator).to(device)
                x0[-damage_n:] *= (1.0 - dmg).unsqueeze(1)
            if noise_n and cfg.noise_std > 0:
                # Gaussian state corruption on every channel: the unroll then has to
                # pull a noisy / partly-destroyed pattern back to the target, which
                # is the robustness pressure the circle damage does not provide
                # (it only ever zeroes contiguous blobs).
                lo = cfg.batch_size - damage_n - noise_n
                hi = cfg.batch_size - damage_n
                noise = torch.randn(
                    (noise_n, x0.shape[1], h, w), device=device, generator=gpu_generator
                ) * cfg.noise_std
                x0[lo:hi] = x0[lo:hi] + noise
        else:
            x0 = seed.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1).contiguous()
            if cond_n:
                sel = torch.arange(cfg.batch_size, device=device) % cond_n
                cond = F.one_hot(sel, cond_n).float()
                tgt = target[sel]
            else:
                cond = None

        n = int(
            torch.randint(cfg.train_steps_min, cfg.train_steps_max, (1,), generator=generator)
        )
        # Draw the whole unroll's fire mask in one shot: one RNG kernel instead
        # of n, and the step becomes deterministic given its inputs.
        masks = (
            torch.rand((n, cfg.batch_size, 1, h, w), device=device, generator=gpu_generator)
            <= ca.fire_rate
        )

        x, loss_t = _train_step(ca, forward, optimizer, x0, tgt, cfg, masks, cond)
        scheduler.step()

        if pool is not None:
            pool.commit(idx, x, cond)

        pending.append(loss_t)

        if step % cfg.log_every == 0 or step == cfg.iterations - 1:
            flush_losses()
            loss = losses[-1]
            elapsed = time.time() - start
            done = step + 1 - start_step
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (cfg.iterations - step - 1) / rate if rate > 0 else float("inf")
            print(
                f"step {step + 1:6d}/{cfg.iterations} | loss {loss:.6f} "
                f"| log10 {np.log10(max(loss, 1e-12)):+.3f} "
                f"| {rate:.2f} it/s | eta {eta / 60:.1f} min",
                flush=True,
            )

        if cfg.progress_every and step % cfg.progress_every == 0:
            rows = [
                tile(render(x0), ncols=cfg.batch_size),
                tile(render(x), ncols=cfg.batch_size),
            ]
            if cond_n:
                # third row: the target each column is being asked to grow
                rows.append(tile(render(tgt), ncols=cfg.batch_size))
            save_image(np.vstack(rows), out_dir / "progress" / f"batch_{step:06d}.png")

        if cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
            flush_losses()
            torch.save(ca.state_dict(), out_dir / f"ckpt_{step + 1:06d}.pt")
            _save_resume_state(
                out_dir / "state.pt", ca, optimizer, scheduler, generator,
                gpu_generator, device, step + 1, losses, pool, cfg.save_pool,
            )
            (out_dir / "loss.json").write_text(json.dumps(losses))

    torch.save(ca.state_dict(), out_dir / "ckpt_final.pt")
    flush_losses()
    _save_resume_state(
        out_dir / "state.pt", ca, optimizer, scheduler, generator,
        gpu_generator, device, cfg.iterations, losses, pool, cfg.save_pool,
    )
    (out_dir / "loss.json").write_text(json.dumps(losses))
    plot_loss(losses, out_dir / "loss.png")

    if cfg.make_gif:
        print("Rendering growth animation...", flush=True)
        cond0 = None
        if cond_n:
            cond0 = F.one_hot(torch.tensor(0, device=device), cond_n).float().unsqueeze(0)
        frames = growth_frames(
            ca,
            seed,
            steps=cfg.gif_steps,
            device=device,
            frame_every=cfg.gif_frame_every,
            scale=cfg.gif_scale,
            cond=cond0,
        )
        save_gif(frames, out_dir / "growth.gif")

    print(f"Done in {(time.time() - start) / 60:.1f} min. Outputs in {out_dir}")
    return losses
