"""Submit a sweep file to Slurm: one TOML in, N sbatch calls out.

    uv run python -m nca.sweep slurm/sweeps/emoji-sweep.toml --dry-run
    bash slurm/submit_sweep.sh slurm/sweeps/emoji-sweep.toml

Replaces `slurm/submit_{emoji,capacity,multitarget,crosstarget}_sweep.sh`, which
were the same 45 lines of sbatch plumbing around a different list of runs. The
run list now lives in the TOML (parsed by :func:`nca.experiments.load_sweep`)
and the placement in `[resources]`, still overridable from the environment so the
old habits keep working:

    PARTITION=mig-priority CONSTRAINT=mig GRES=gpu:1g.10gb:1 \
        bash slurm/submit_sweep.sh slurm/sweeps/cross.toml

Each job is `slurm/train.sbatch --config <toml> --run <name>`, so the sweep file
is the only argument that ever changes -- and the job name in `squeue` is the
model's id.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace

from .experiments import Experiment, Resources, Sweep, load_sweep

# Environment overrides, named as the old shell scripts named them.
ENV_OVERRIDES = {
    "partition": "PARTITION",
    "constraint": "CONSTRAINT",
    "gres": "GRES",
    "time": "TIME",
    "mem": "MEM",
    "cpus": "CPUS",
}


def resources_from_env(resources: Resources) -> Resources:
    updates = {}
    for key, env in ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if value:
            updates[key] = int(value) if key == "cpus" else value
    return replace(resources, **updates) if updates else resources


def sbatch_command(
    sweep: Sweep, run: Experiment, resources: Resources, *, config: str, depend: str = ""
) -> list[str]:
    """The argv for one run: placement flags, then what the job should train."""
    cmd = [
        "sbatch",
        f"--job-name={run.name}",
        f"--partition={resources.partition}",
    ]
    if resources.constraint:
        cmd.append(f"--constraint={resources.constraint}")
    cmd += [
        f"--gres={resources.gres}",
        f"--time={resources.time}",
        f"--mem={resources.mem}",
        f"--cpus-per-task={resources.cpus}",
    ]
    if depend:
        cmd.append(f"--dependency=afterany:{depend}")
    cmd += ["slurm/train.sbatch", "--config", config, "--run", run.name]
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sweep", help="path to slurm/sweeps/*.toml")
    p.add_argument("--only", nargs="+", metavar="RUN", help="submit only these run names")
    p.add_argument("--dry-run", action="store_true", help="print the sbatch lines, submit nothing")
    p.add_argument(
        "--depend",
        default=os.environ.get("DEPEND", ""),
        help="comma-separated job ids this sweep must run after (afterany)",
    )
    args = p.parse_args(argv)

    sweep = load_sweep(args.sweep)
    resources = resources_from_env(sweep.resources)
    runs = sweep.runs
    if args.only:
        unknown = sorted(set(args.only) - {r.name for r in sweep.runs})
        if unknown:
            p.error(f"unknown run(s) {unknown}; have {[r.name for r in sweep.runs]}")
        runs = tuple(r for r in sweep.runs if r.name in set(args.only))
    depend = args.depend.replace(",", ":")

    print(f"{sweep.name}: {len(runs)} run(s) -> {resources.partition} {resources.constraint}")
    failed = 0
    for run in runs:
        cmd = sbatch_command(
            sweep, run, resources, config=os.path.abspath(args.sweep), depend=depend
        )
        if args.dry_run:
            print(" ".join(cmd))
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = (proc.stdout or proc.stderr).strip().splitlines()
        status = "ok  " if proc.returncode == 0 else "FAIL"
        print(f"  {status} {run.name:<32} {(line[0] if line else '').strip()}")
        failed += proc.returncode != 0
    if args.dry_run:
        return 0
    print(f"{len(runs) - failed} submitted, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
