"""``ModelSpec``: the single source of truth for a CA model's shape.

Everything that needs to know "how big is this model" -- the module, the ONNX
exporter, the ``weights.bin`` layout, the WGSL override constants, the
frontend's type declarations -- should read it from here instead of inferring it
from a state dict or re-deriving it by hand.

This module is deliberately **torch-free**: it only inspects state-dict keys and
tensor shapes through duck typing, so the headless shader harness
(``scripts/shader_bench.py``) and any tooling without torch can import it.
Anything that needs to *build* a module does ``spec.build()``, which imports
torch lazily.

The one rule: **read a model's shape, never infer it.** ``infer_dims`` used to
be duplicated in ``nca/export`` and ``nca/sample.py``; the copy in
``sample.py`` computed ``in_channels // 3``, which is wrong for any conditional
model (``3C + cond_n``) and blind to ``hidden_layers`` / ``learn_perception``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

#: Channels 0..2 are RGB and channel 3 is alpha, so 4 is the minimum.
MIN_CHANNELS = 4

#: Number of floats in the learned (dx, dy) Sobel pair of a ``learn_perception``
#: model: 2 kernels x 3 x 3.
SOBEL_FLOATS = 18


def _conv_indices(state_dict: Mapping[str, Any]) -> list[int]:
    """Sorted indices ``N`` of every ``update.N.weight`` in a state dict."""
    idx = [
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("update.") and k.endswith(".weight")
    ]
    if not idx:
        raise ValueError("state dict has no update.*.weight entries")
    return sorted(idx)


def _last_conv_index(state_dict: Mapping[str, Any]) -> int:
    """Index of the final ``update.N`` (the output projection)."""
    return max(_conv_indices(state_dict))


def _hidden_layers(state_dict: Mapping[str, Any]) -> int:
    """Number of hidden (ReLU) blocks: convs minus the output projection."""
    return max(len(_conv_indices(state_dict)) - 1, 1)


@dataclass(frozen=True)
class ModelSpec:
    """The shape (and a few non-tensor hyper-parameters) of one CA model.

    Frozen and hashable so it can be used as a cache key (e.g. for WGSL pipeline
    specialisation) and so nobody mutates it in place.
    """

    channel_n: int = 16
    hidden_n: int = 128
    hidden_layers: int = 1
    cond_n: int = 0
    learn_perception: bool = False
    wrap: bool = True
    fire_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.channel_n < MIN_CHANNELS:
            raise ValueError(
                f"channel_n must be >= {MIN_CHANNELS} (RGB + alpha), got {self.channel_n}"
            )
        if self.hidden_n <= 0:
            raise ValueError(f"hidden_n must be positive, got {self.hidden_n}")
        if self.hidden_n % 4:
            # The shader loads the MLP in vec4 rows (`HG = hidden/4`) and the
            # exporter pads W1 rows to a multiple of 4; a hidden width that is
            # not a multiple of 4 would silently drop the tail units on the GPU.
            raise ValueError(
                f"hidden_n must be a multiple of 4 (the WGSL step reads vec4 rows), got {self.hidden_n}"
            )
        if self.hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {self.hidden_layers}")
        if self.cond_n < 0:
            raise ValueError(f"cond_n must be >= 0, got {self.cond_n}")

    # --- derived quantities ------------------------------------------------

    @property
    def conv_n(self) -> int:
        """Channels out of the perception conv: ``3C`` (identity, d/dx, d/dy).

        This is the last axis of the ``perception_weight`` graph input. It is
        *not* :attr:`perception_n` -- the ``cond_n`` one-hot is concatenated to the
        conv output, it never enters the convolution.
        """
        return 3 * self.channel_n

    @property
    def perception_n(self) -> int:
        """Fan-in of ``update[0]``: ``3C`` (identity, dx, dy) plus the selector."""
        return self.conv_n + self.cond_n

    @property
    def perception_padded_n(self) -> int:
        """``perception_n`` rounded up to a multiple of 4 for ``vec4`` weight loads."""
        return (self.perception_n + 3) // 4 * 4

    @property
    def hidden_groups(self) -> int:
        """``hidden_n / 4`` -- the WGSL ``HG`` override."""
        return self.hidden_n // 4

    @property
    def is_conditional(self) -> bool:
        return self.cond_n > 0

    @property
    def weights_padding(self) -> int:
        """Zero floats added to ``weights.bin`` that are **not** parameters.

        Each ``W1`` row is padded from ``perception_n`` to ``perception_padded_n``
        so the shader can load it as ``vec4``; those columns are not parameters.
        Non-zero only when ``perception_n % 4 != 0`` (e.g. ``C=6`` -> 2 per row).
        """
        return self.hidden_n * (self.perception_padded_n - self.perception_n)

    def num_parameters(self) -> int:
        """Parameter count implied by the spec (independent of any module).

        Used both for the meta block and as a consistency check against
        ``NeuralCA.num_parameters``: if they disagree, a layer is unaccounted for.
        """
        p = self.perception_n
        h = self.hidden_n
        total = h * p + h  # update[0]
        total += (self.hidden_layers - 1) * h * (h + 1)  # hidden -> hidden blocks
        total += self.channel_n * h + self.channel_n  # output projection
        if self.learn_perception:
            total += SOBEL_FLOATS
        return total

    # --- weights.bin layout ------------------------------------------------

    def weight_offsets(self) -> dict[str, int]:
        """Offsets (in floats) of each block inside ``weights.bin``.

        Layout for ``L`` hidden layers::

            W1 [hidden, P4], b1 [hidden],
            (W_i [hidden, hidden], b_i [hidden]) for i = 2..L,
            W_out [C, hidden], b_out [C]

        Key order matters only for readability; the values must match
        ``NeuralCA``'s parameter order, which is what ``save_weights`` writes.
        """
        h = self.hidden_n
        p4 = self.perception_padded_n
        w1 = h * p4
        off = {"w1": 0, "b1": w1}
        cursor = w1 + h
        if self.hidden_layers > 1:
            # One block per intermediate layer: W (hidden*hidden) then b (hidden).
            off["wmid"] = cursor
            off["bmid"] = cursor + h * h
            off["mid_stride"] = h * h + h
            cursor += off["mid_stride"] * (self.hidden_layers - 1)
        off["w2"] = cursor
        off["b2"] = cursor + self.channel_n * h
        return off

    def weights_count(self) -> int:
        """Total float count of ``weights.bin`` (padding included)."""
        return self.weight_offsets()["b2"] + self.channel_n

    def state_from_weights(self, blob: Any) -> dict[str, Any]:
        """Decode a ``weights.bin`` blob back into the update rule's parameters.

        The exact inverse of :meth:`weight_offsets` (plus the ``W1`` row padding),
        and deliberately the *only* implementation of that inverse: an export is
        supposed to be a complete description of its model, and this is what makes
        that true after the training run is gone.

        Returns numpy arrays keyed like ``NeuralCA.state_dict()`` (``update.N.*``);
        the caller wraps them in torch. Raises if the padding is not zero, because
        that means the blob was not written by :func:`nca.export.weights.save_weights`.
        """
        import numpy as np

        blob = np.asarray(blob, dtype="<f4").reshape(-1)
        if blob.size != self.weights_count():
            raise ValueError(
                f"weights.bin has {blob.size} floats, spec says {self.weights_count()} "
                f"({self.channel_n} channels, hidden {self.hidden_n}, "
                f"{self.hidden_layers} layers, cond_n {self.cond_n})"
            )
        h, p, p4, c = self.hidden_n, self.perception_n, self.perception_padded_n, self.channel_n
        off = self.weight_offsets()
        w1 = blob[off["w1"] : off["w1"] + h * p4].reshape(h, p4)
        if p4 != p and np.any(w1[:, p:] != 0):
            raise ValueError(
                f"weights.bin row padding is not zero ({p4 - p} floats per row): "
                "this blob was not written by save_weights, so it cannot be decoded"
            )
        last = 2 * self.hidden_layers
        state: dict[str, Any] = {
            "update.0.weight": w1[:, :p].reshape(h, p, 1, 1).copy(),
            "update.0.bias": blob[off["b1"] : off["b1"] + h].copy(),
        }
        for li in range(1, self.hidden_layers):
            base = off["wmid"] + (li - 1) * off["mid_stride"]
            state[f"update.{2 * li}.weight"] = blob[base : base + h * h].reshape(h, h, 1, 1).copy()
            state[f"update.{2 * li}.bias"] = blob[base + h * h : base + h * h + h].copy()
        state[f"update.{last}.weight"] = blob[off["w2"] : off["w2"] + c * h].reshape(c, h, 1, 1).copy()
        state[f"update.{last}.bias"] = blob[off["b2"] : off["b2"] + c].copy()
        return state

    def weights_meta(self, file: str = "weights.bin") -> dict[str, Any]:
        """The ``meta.json`` ``weights`` block for this spec."""
        return {
            "file": file,
            "channel_n": self.channel_n,
            "hidden_n": self.hidden_n,
            "hidden_layers": self.hidden_layers,
            "perception_n": self.perception_n,
            "perception_padded_n": self.perception_padded_n,
            "offsets": self.weight_offsets(),
            "count": self.weights_count(),
        }

    def meta_fields(self) -> dict[str, Any]:
        """The spec-owned subset of a ``meta.json`` (flat, for compatibility)."""
        return {
            "channel_n": self.channel_n,
            "hidden_n": self.hidden_n,
            "hidden_layers": self.hidden_layers,
            "cond_n": self.cond_n,
            "fire_rate": self.fire_rate,
            "learn_perception": self.learn_perception,
            "wrap": self.wrap,
            "num_parameters": self.num_parameters(),
        }

    # --- construction ------------------------------------------------------

    def build(self):
        """Build a ``NeuralCA`` for this spec (imports torch lazily)."""
        from .model import NeuralCA

        return NeuralCA.from_spec(self)

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, Any], *, wrap: bool = True
    ) -> "ModelSpec":
        """Recover the spec from a checkpoint's tensors.

        ``channel_n`` comes from the *output* projection: ``update.0``'s fan-in is
        ``3C + cond_n``, so deriving ``C`` from it (as ``nca/sample.py`` used to)
        silently breaks every conditional model.

        ``wrap`` is not stored in a state dict, so the caller supplies it.
        """
        hidden_n = int(state_dict["update.0.weight"].shape[0])
        channel_n = int(state_dict[f"update.{_last_conv_index(state_dict)}.weight"].shape[0])
        fan_in = int(state_dict["update.0.weight"].shape[1])
        cond_n = fan_in - 3 * channel_n
        if cond_n < 0:
            raise ValueError(
                f"update.0 fan-in {fan_in} is too small for channel_n={channel_n}"
            )
        return cls(
            channel_n=channel_n,
            hidden_n=hidden_n,
            hidden_layers=_hidden_layers(state_dict),
            cond_n=cond_n,
            learn_perception="sobel" in state_dict,
            wrap=wrap,
        )

    @classmethod
    def from_model(cls, ca: Any) -> "ModelSpec":
        """Recover the spec from a live ``NeuralCA`` (duck-typed)."""
        spec = cls.from_state_dict(ca.state_dict(), wrap=bool(ca.wrap))
        return replace(spec, fire_rate=float(ca.fire_rate))

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> "ModelSpec":
        """Recover the spec from an exported ``meta.json``."""
        weights = meta.get("weights") or {}
        return cls(
            channel_n=int(meta["channel_n"]),
            hidden_n=int(meta["hidden_n"]),
            hidden_layers=int(meta.get("hidden_layers") or weights.get("hidden_layers") or 1),
            cond_n=int(meta.get("cond_n") or 0),
            learn_perception=bool(meta.get("learn_perception", False)),
            wrap=bool(meta.get("wrap", True)),
            fire_rate=float(meta.get("fire_rate", 0.5)),
        )

    @classmethod
    def from_run_dir(cls, path: str | Path) -> "ModelSpec":
        """Load the spec of a training run.

        Prefers ``spec.json`` (written by ``train.py``); falls back to reading the
        final checkpoint, which is what pre-``spec.json`` runs have.
        """
        path = Path(path)
        spec_file = path / "spec.json"
        if spec_file.exists():
            spec = cls.from_json(json.loads(spec_file.read_text()))
            cls._check_against_checkpoint(spec, find_checkpoint(path))
            return spec
        checkpoint = find_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError(f"no spec.json or checkpoint in {path}")
        return cls.from_checkpoint(checkpoint)

    @classmethod
    def _check_against_checkpoint(cls, spec: "ModelSpec", checkpoint: Path | None) -> None:
        """Refuse a ``spec.json`` that disagrees with the weights next to it.

        ``train.py`` writes ``spec.json`` before training resumes, so an aborted
        resume could leave a spec that describes a different module than the
        checkpoints in the same directory. Reading the shape from the checkpoint is
        the only way to notice, and a wrong spec is worse than no spec: every
        consumer (sample, export, the shader override constants) trusts it.
        """
        if checkpoint is None:
            return
        actual = cls.from_state_dict(load_model_state(checkpoint))
        for field in ("channel_n", "hidden_n", "hidden_layers", "cond_n", "learn_perception"):
            want, got = getattr(actual, field), getattr(spec, field)
            if want != got:
                raise ValueError(
                    f"{checkpoint.parent}/spec.json says {field}={got!r} but "
                    f"{checkpoint.name} has {field}={want!r}; the run directory is "
                    f"inconsistent (an aborted resume?). Re-derive the spec with "
                    f"ModelSpec.from_checkpoint()."
                )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, wrap: bool = True) -> "ModelSpec":
        """Spec of a ``.pt`` checkpoint, with non-tensor fields from its metadata.

        A state dict carries no ``fire_rate``/``wrap``, so those come from a sibling
        ``spec.json``/``meta.json`` when there is one (otherwise the dataclass
        defaults, which is what older checkpoints were trained with).
        """
        checkpoint = Path(checkpoint)
        spec = cls.from_state_dict(load_model_state(checkpoint), wrap=wrap)
        for name in ("spec.json", "meta.json"):
            side = checkpoint.parent / name
            if not side.exists():
                continue
            try:
                data = json.loads(side.read_text())
            except json.JSONDecodeError:
                continue
            if "fire_rate" in data:
                spec = replace(spec, fire_rate=float(data["fire_rate"]))
            if wrap and "wrap" in data:
                spec = replace(spec, wrap=bool(data["wrap"]))
            break
        return spec

    # --- serialisation -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}  # ignore unknown keys
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:  # a compact one-liner for logs
        bits = [f"C={self.channel_n}", f"h={self.hidden_n}"]
        if self.hidden_layers != 1:
            bits.append(f"L={self.hidden_layers}")
        if self.cond_n:
            bits.append(f"cond={self.cond_n}")
        if self.learn_perception:
            bits.append("perc=learned")
        if not self.wrap:
            bits.append("wrap=off")
        return " ".join(bits)


def load_model_state(checkpoint: str | Path) -> Mapping[str, Any]:
    """Load a ``.pt`` and unwrap it to a bare model state dict.

    A training checkpoint is either a bare ``state_dict`` (``ckpt_*.pt``) or a
    resume blob with the model under ``"model"`` (``state.pt``). This is the one
    place that knows that difference.
    """
    import torch

    state = torch.load(checkpoint, map_location="cpu")
    if "model" in state and "update.0.weight" not in state:
        state = state["model"]
    return state


def find_checkpoint(run_dir: str | Path) -> Path | None:
    """The checkpoint worth exporting from a run directory.

    ``ckpt_final.pt`` wins; otherwise the highest-numbered ``ckpt_NNNNNN.pt``
    (compared numerically -- ``sorted()`` on the names is lexicographic and picks
    the wrong one past six digits).
    """
    run_dir = Path(run_dir)
    final = run_dir / "ckpt_final.pt"
    if final.exists():
        return final
    numbered = list(run_dir.glob("ckpt_*.pt"))
    if not numbered:
        return None

    def step(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else -1

    return max(numbered, key=step)
