"""Score the 13 perceptual-sweep checkpoints on measures none of them optimised.

Grow from one seed on each run's own 8-target canvas, then keep stepping (no
fire mask) and see how much the picture moves: that drift is what the sweep was
supposed to fix, and it is invisible in a per-run loss curve because every run
optimised a different objective.
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nca.spec import ModelSpec, load_model_state  # noqa: E402
from nca.step import make_seed  # noqa: E402
from scripts.loss_ab import _metrics  # noqa: E402

GROW = 96
COAST = 200


def score(run: pathlib.Path) -> dict:
    spec = ModelSpec.from_run_dir(run)
    ca = spec.build()
    ca.load_state_dict(load_model_state(run / "ckpt_final.pt"))
    ca.eval()

    target = torch.from_numpy(np.load(run / "target.npy")).float()
    n, _, h, w = target.shape
    seed = torch.from_numpy(make_seed(spec.channel_n, h, w)).float()
    seed = seed.unsqueeze(0).repeat(n, 1, 1, 1)

    # A conditional model (multi-target) gets its own one-hot per canvas row.
    cond = None
    if spec.cond_n:
        cond = torch.eye(spec.cond_n)[:n].reshape(n, spec.cond_n, 1, 1)

    with torch.no_grad():
        x = seed
        for _ in range(GROW):
            x = ca(x, cond=cond)
        grown = _mean_metrics(x, target)
        for _ in range(COAST):
            x = ca(x, cond=cond)
        coasted = _mean_metrics(x, target)
    return {
        "grown": grown,
        "coast": coasted,
        "drift": {
            k: coasted[k] - grown[k] for k in ("psnr", "ssim", "edge", "alpha")
        },
    }


def _mean_metrics(state: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rows = [_metrics(state[i : i + 1], target[i : i + 1]) for i in range(target.shape[0])]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    runs = sorted((root / "runs").glob("cross8.mismatch.h=128.perc.*"))
    out = {}
    print(
        f"{'variant':12s} | {'grown: psnr  ssim  edge  iou':30s} | "
        f"{'after 200 idle: psnr  ssim  edge  iou':34s} | drift"
    )
    for run in runs:
        s = score(run)
        g, c, d = s["grown"], s["coast"], s["drift"]
        name = run.name.split(".perc.")[-1]
        out[name] = s
        print(
            f"{name:12s} | {g['psnr']:5.2f} {g['ssim']:.3f} {g['edge']:.3f} {g['alpha']:.3f}       | "
            f"{c['psnr']:5.2f} {c['ssim']:.3f} {c['edge']:.3f} {c['alpha']:.3f}            | "
            f"iou {d['alpha']:+.3f} edge {d['edge']:+.3f}",
            flush=True,
        )
    (root / "runs" / "perc-sweep-eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
