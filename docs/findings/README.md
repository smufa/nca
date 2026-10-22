# Findings

Measured results, with the reasoning that produced them. One file per question.

| file | answers |
|---|---|
| [`shader-perf.md`](shader-perf.md) | why the step costs what it costs on the GPU, what was tried and rejected, what is left to try |
| [`training-perf.md`](training-perf.md) | batch/AMP/compile scaling, memory, H100 vs MIG |
| [`capacity.md`](capacity.md) | how loss scales with channels/hidden width |
| [`robustness.md`](robustness.md) | why the kernel lab breaks a plain model, and what training regime fixes it |
| [`conditioning.md`](conditioning.md) | multi-target models: capacity, `--mismatch-p`, wide vs deep, what the playground can show |
| [`frontend-engines.md`](frontend-engines.md) | why there are two engines, why the ORT WebGPU EP was removed, what the brush/sync contract is |

## Policy

- **Append-only.** When a measurement is superseded, add the new number and mark
  the old one `[SUPERSEDED by …]` with what changed. Do not silently rewrite
  history: a wrong number that was once believed is how a wrong decision gets
  re-made.
- **Every number must be regenerable by a command**, and the command goes in the
  file. If it cannot be regenerated, say so and say why (e.g. a `/tmp` checkpoint
  is gone).
- **Losses are not comparable across regimes.** A `--noise-n` run and a plain run
  optimise different objectives; only compare within one regime, or compare the
  behaviour you actually care about (the playground, `nca.sample`, a diff table).
- **Say what it cost.** A finding that says "depth wins" without the 38% parameter
  and 10.2 ms/step cost is not useful when the next experiment has a different
  budget.
