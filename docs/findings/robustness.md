# Robustness: the kernel lab is a probe, not a toy

Perturbing the perception kernels in the playground's kernel lab gives a cell a
scrambled view of its neighbourhood **without changing its state**. A model trained
only on clean perception + circle damage has no return path from the states that
produces: it falls into a noisy attractor that is absorbing and never recovers.
That is an out-of-distribution perturbation the training regime never covered.

Training with state noise (`--experiment noisy`, or `--noise-n` / `--noise-std`)
fixes it: the update rule is then asked to map a whole neighbourhood of corrupted
states back to the target, which widens the target's basin. It costs nothing in
fidelity — `crab_noisy` reaches min loss **2.45e-4** vs **2.58e-4** for the plain
fp32 `regenerating` baseline, with 4 of every 16 pool samples corrupted.

## Takeaway

Robustness tracks the **breadth of perturbation seen during training**, not the
amount of damage. Circle damage only ever zeroes contiguous blobs, so it does not
cover the space the kernel lab can reach. Neither is captured by the training loss,
which is not comparable across regimes — the failure was found by playing with the
kernel lab, not by a metric.

Reproduce: `bash scripts/bench.sh crab_noisy --reference` for the shader path, or
grow both models in the playground and drag the kernel sliders.
