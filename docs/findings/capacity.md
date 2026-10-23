# Model capacity (crab / regenerating, 8k iters)

The paper size (C=16, hidden=128, **8336 params**) reaches min loss **2.0e-4**. The
sweep models are the `exports/cap.c=*.h=*` entries; regenerate with
`bash slurm/submit_sweep.sh slurm/sweeps/capacity.toml`.

| hidden | C | params | min loss | ×base |
|---:|---:|---:|---:|---:|
| 128 | 16 | 8336 | 2.0e-4 | 1 × |
| 128 | 12 | 6284 | 2.4e-3 | 12 × |
| 128 | 8 | 4232 | 5.7e-3 | 28 × |
| 64 | 16 | 4176 | 8.8e-3 | 44 × |
| 128 | 6 | 3206 | 9.2e-3 | 46 × |
| 32 | 16 | 2096 | 1.3e-2 | 63 × |
| 16 | 16 | 1056 | 2.0e-2 | 99 × |
| 8 | 16 | 536 | 1.9e-2 | 96 × |

## Conclusions

- **Channels are the cheaper axis to cut at equal parameters**: C=12 (6284 params,
  2.4e-3) beats hidden=64 (4176 params, 8.8e-3) and even hidden=128 at C=8.
- **Below ~4k params quality falls off a cliff** (5.7e-3 → 9.2e-3 → 2.0e-2 for the
  last four rows, i.e. 1.6×→3.6×→10× the best).
- The bottom two rows are not ordered the way the parameter count suggests
  (536 params scores slightly better than 1056), which is noise at this
  compression level: both are far outside the usable regime.
- A `C=` / `hidden=` change is a **shader** change too: `C` drives `NP4`/`W1S4` and
  `hidden` drives `HG`, both compile-time overrides (see `shader-perf.md`). All of
  the supported cells here (4 ≤ C ≤ 16, hidden % 4 == 0) run natively.
