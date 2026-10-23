"""How far does the CA drift if the step is computed in fp16?

Simulates the three ways fp16 would enter the kernel, using the real weights of an
exported model and the same fire masks for every variant:

  weights   W1/W2/b held as fp16, everything else fp32 (fp32 accumulate)
  state     the state tensor rounded to fp16 after every step
  acts      perception + hidden activations rounded to fp16 (the MLP itself fp16)
  all       weights + state + acts

The reference is the fp32 step the ONNX graph computes (same maths, one extra
float32 rounding of the perception).

Usage: .venv/bin/python scripts/fp16_drift.py exports/crab.regenerating --steps 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nca.step import make_seed, perception_weight  # noqa: E402  (scripts/ is not a package)


def f16(a: np.ndarray) -> np.ndarray:
    """Round to IEEE fp16 and back to fp32: exactly 'stored in half precision'."""
    return a.astype(np.float16).astype(np.float32)


def int8(a: np.ndarray) -> np.ndarray:
    """Symmetric per-row int8 quantisation, dequantised.

    Per output row (a neuron), which is the layout an int8 kernel wants: one scale
    per row of `W1`/`W2`, one per bias vector. Dequantised back to f32 because the
    question here is accuracy, not speed.
    """
    a = np.asarray(a, np.float32)
    if a.ndim == 1:
        scale = max(float(np.abs(a).max()) / 127.0, 1e-12)
        return np.round(a / scale) * scale
    scale = np.maximum(np.abs(a).max(axis=1, keepdims=True) / 127.0, 1e-12)
    return np.round(a / scale) * scale


def load(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    w = np.fromfile(model_dir / "weights.bin", dtype="<f4")
    off = meta["weights"]["offsets"]
    C, H = int(meta["channel_n"]), int(meta["weights"]["hidden_n"])
    hidden_layers = int(meta["weights"].get("hidden_layers", 1))
    n_perc = int(meta["weights"]["perception_n"])
    perc_pad = int(meta["weights"]["perception_padded_n"])

    def block(start, n):
        return w[start : start + n].reshape(-1)

    W1 = block(off["w1"], H * perc_pad).reshape(H, perc_pad)
    b1 = block(off["b1"], H)
    W2 = block(off["w2"], C * H).reshape(C, H)
    b2 = block(off["b2"], C)
    mids = []
    if hidden_layers > 1:
        cur = off["b2"] + C
        for _ in range(hidden_layers - 1):
            mids.append(block(cur, H * H).reshape(H, H))
            cur += H * H
            mids.append(block(cur, H))
            cur += H
    return {
        "meta": meta,
        "C": C,
        "H": H,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "mids": mids,
        "n_perc": n_perc,
        "perc_pad": perc_pad,
    }


def make_step(w, grid: int, rounding: str, noise: float = 0.0):
    C, WH, n_perc, pad = w["C"], w["H"], w["n_perc"], w["perc_pad"]
    KW = perception_weight(C)
    W1, b1, W2, b2, mids = w["W1"], w["b1"], w["W2"], w["b2"], w["mids"]
    if rounding in ("weights", "all"):
        W1, b1, W2, b2 = f16(W1), f16(b1), f16(W2), f16(b2)
        mids = [f16(m) for m in mids]
    if rounding == "int8":
        W1, b1, W2, b2 = int8(W1), int8(b1), int8(W2), int8(b2)
        mids = [int8(m) for m in mids]
    use_acts = rounding in ("acts", "all")
    use_state = rounding in ("state", "all")
    rng = np.random.default_rng(12345)

    def perception(x):
        # The shared basis from nca/step.py (identity, Sobel-x/8, Sobel-y/8) applied
        # depthwise per channel, wrapped: the same 9 taps the WGSL kernel and the
        # ONNX Conv see.
        p = np.zeros((x.shape[0], n_perc, *x.shape[2:]), np.float32)
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="wrap")
        for c in range(C):
            for k in range(3):
                w = KW[c * 3 + k, 0]
                acc = p[:, c * 3 + k]
                for dy in range(3):
                    for dx in range(3):
                        if w[dy, dx]:
                            acc += xp[:, c, dy : dy + grid, dx : dx + grid] * w[dy, dx]
        return p

    def step(x, fire):
        p = perception(x)
        if pad > n_perc:  # conditional models pad the perception vector
            p = np.concatenate([p, np.zeros((x.shape[0], pad - n_perc, *p.shape[2:]), np.float32)], axis=1)
        a = p.transpose(0, 2, 3, 1) @ W1.T + b1  # [B,H,W,hidden]
        if use_acts:
            a = f16(a)
        h = np.maximum(a, 0.0)
        for i in range(0, len(mids), 2):
            h = np.maximum(h @ mids[i].T + mids[i + 1], 0.0)
            if use_acts:
                h = f16(h)
        out = h @ W2.T + b2  # [B,H,W,C]
        out = out.transpose(0, 3, 1, 2)
        nxt = x + out * fire  # fire: [B,1,H,W]
        if noise:  # a fp32-rounding analogue: perturb by one ulp*scale
            nxt = nxt + rng.normal(0.0, noise, nxt.shape).astype(np.float32)
        if use_state:
            nxt = f16(nxt)
        return nxt

    return step


def living_mask(x):
    """pass2.wgsl: a cell is alive if alpha > 0.1 anywhere in its 3x3 neighbourhood."""
    a = x[0, 3]  # [H,W]
    a = np.pad(a, 1, mode="wrap")
    win = np.lib.stride_tricks.sliding_window_view(a, (3, 3)).max(axis=(2, 3))
    return (win > 0.1).astype(np.float32)[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", type=str, default=None,
                    help="comma list of eps,weights,acts,state,all,int8 (default: all but state)")
    ap.add_argument("--no-living", action="store_true",
                    help="skip the pass2 living mask (what the shader bench compares, and "
                         "what makes long-run drift saturate)")
    ap.add_argument("--heal", type=int, default=0,
                    help="after the run, damage a circle and let both heal this many "
                    "steps, then report the healed-visual difference")
    args = ap.parse_args()

    w = load(args.model)
    grid = args.grid or min(int(w["meta"]["grid"]), 72)
    C = w["C"]
    rng = np.random.default_rng(args.seed)

    # Same initial condition for every variant: the seed cell from nca/step.py.
    x0 = make_seed(C, grid, grid)[None].astype(np.float32)

    fire_rate = float(w["meta"].get("fire_rate", 0.5))
    masks = (rng.random((args.steps, 1, grid, grid)) < (1.0 - fire_rate)).astype(np.float32)

    # "eps" is the control: fp32 arithmetic with a 1e-7 state perturbation, i.e. the
    # size of a float32 rounding difference. If fp16 diverges like eps does, the
    # divergence is chaos amplifying rounding noise, not fp16 error.
    variants = [v for v in (args.variants or "eps,weights,acts,all,int8").split(",") if v]
    steps = {v: make_step(w, grid, "none" if v == "eps" else v, noise=1e-7 if v == "eps" else 0.0)
             for v in variants}
    ref_step = make_step(w, grid, "none")

    ref = x0.copy()
    run = {v: x0.copy() for v in variants}

    print(f"model {args.model.name} · grid {grid}² · C={C} hidden={w['H']} · "
          f"fire_rate {fire_rate} · {args.steps} steps")
    print(f"{'step':>5} " + " ".join(f"{v + ' max':>12}" for v in variants)
          + "   " + " ".join(f"{v + ' IoU':>9}" for v in variants))
    first = {v: {} for v in variants}  # threshold -> first step it is crossed

    for t in range(args.steps):
        f = masks[t]
        prev = ref.copy()
        ref = ref_step(ref, f)
        # living mask: a cell survives if alpha > 0.1 in its 3x3 neighbourhood, before
        # and after the update (pass2.wgsl)
        alive = np.ones((1, 1, grid, grid), np.float32)
        if not args.no_living:
            alive = np.minimum(living_mask(prev), living_mask(ref))
        ref = ref * alive  # pass2.wgsl multiplies every channel by the living mask
        for v in variants:
            run[v] = steps[v](run[v], f) * alive
        for v in variants:
            d = float(np.abs(run[v] - ref).max())
            for thr in (1e-3, 1e-2, 1e-1, 1.0):
                if d > thr and thr not in first[v]:
                    first[v][thr] = t + 1
        if (t + 1) % 25 == 0 or t == 0:
            maxs, ious = [], []
            for v in variants:
                d = np.abs(run[v] - ref).max()
                maxs.append(d)
                a, b = (run[v][0, 3] > 0.1), (ref[0, 3] > 0.1)
                ious.append((a & b).sum() / max(1, (a | b).sum()))
            print(f"{t + 1:5d} " + " ".join(f"{m:12.3e}" for m in maxs)
                  + "   " + " ".join(f"{i:9.4f}" for i in ious))

    print()
    for v in variants:
        d = np.abs(run[v] - ref)
        print(f"{v:8s} final per-channel max|Δ|: " +
              " ".join(f"{x:.1e}" for x in d[0].reshape(C, -1).max(axis=1)) +
              f"   (visible rgb+alpha {d[0, :4].max():.1e})")
    print()
    print(f"{'variant':8s} " + " ".join(f"{'>' + str(t):>7}" for t in (1e-3, 1e-2, 1e-1, 1.0)))
    for v in variants:
        cells = " ".join(f"{first[v].get(t, '-'):>7}" for t in (1e-3, 1e-2, 1e-1, 1.0))
        print(f"{v:8s} {cells}")
    print()

    # What the eye sees: composite over white, compare with the fp32 render.
    def render(x):
        rgb = x[0, :3]
        a = x[0, 3:4]
        return np.clip(rgb + (1.0 - a), 0.0, 1.0)

    ref_img = render(ref)
    for v in variants:
        img = render(run[v])
        mse = float(((img - ref_img) ** 2).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        alpha_ref, alpha_v = ref[0, 3] > 0.1, run[v][0, 3] > 0.1
        img8 = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img8).save(Path("/tmp") / f"fp16-{v}-{args.steps}.png")
        if args.heal:
            # Regeneration test: the same damage applied to both, then healed in its
            # own precision. This is the property the model exists for.
            def damaged(x, r=6):
                y = x.copy()
                cy = cx = grid // 2
                yy, xx = np.mgrid[0:grid, 0:grid]
                hole = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
                y[0, :, hole] = 0.0
                return y

            def heal(x, st):
                h = damaged(x)
                for _ in range(args.heal):
                    m = (rng.random((1, 1, grid, grid)) < (1.0 - fire_rate)).astype(np.float32)
                    pv = h
                    h = st(h, m)
                    al = np.minimum(living_mask(pv), living_mask(h))
                    h = h * al
                    if h[0, 3].max() <= 0.1:  # the model needs a live cell to grow from
                        break
                return h

            h_ref = heal(ref, ref_step)
            h_var = heal(run[v], steps[v])
            hi_ref, hi_var = render(h_ref), render(h_var)
            hmse = float(((hi_ref - hi_var) ** 2).mean())
            print(f"         healed render PSNR {10 * np.log10(1 / hmse):5.1f} dB · "
                  f"healed alpha IoU "
                  f"{((h_ref[0, 3] > 0.1) & (h_var[0, 3] > 0.1)).sum() / max(1, ((h_ref[0, 3] > 0.1) | (h_var[0, 3] > 0.1)).sum()):.4f}"
                  + (f" · healed state max|Δ| {np.abs(h_ref - h_var).max():.2e}"))
        print(f"{v:8s} render PSNR {psnr:5.1f} dB · alpha IoU "
              f"{(alpha_ref & alpha_v).sum() / max(1, (alpha_ref | alpha_v).sum()):.4f} · "
              f"state max|Δ| {np.abs(run[v] - ref).max():.2e}")


if __name__ == "__main__":
    main()
