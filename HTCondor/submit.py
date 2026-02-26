#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
from math import ceil


REPO = "/work/omunkombwe/k4RecTracker"
RUN_SCRIPT = f"{REPO}/HTCondor/gun_fullsim_reco.sh"

# Physics setup
P_MIN = 0.2
P_MAX = 10.0
PARTICLE_TAG = "mu"  # used in output file naming, e.g. "muon_P0.2-10GeV_theta10deg_seed1001_nev1000_part1.root"
ANGLES = [10, 20, 30, 40, 46, 60, 75, 89]
TOTAL_EVENTS_PER_ANGLE = 100000
EVENTS_PER_JOB = 1000

# Seed offsets to keep streams separated per angle
SEED_BASE = {
    10: 1000,   
    20: 2000,
    30: 3000,
    40: 4000,
    46: 4600,
    60: 95876,
    75: 6562,
    89: 278,
}

# Condor resource setup
REQUEST_CPUS = 1
REQUEST_MEMORY_MB = 10000
REQUEST_DISK_KB = 1000000
REQUEST_WALLTIME_HOURS = 4

# Output workspace for generated submit scripts/config
WORKDIR = f"{REPO}/HTCondor/gun_split_submit"
OUTPUT_BASE = "/ceph/omunkombwe/mu_Grand"  # Base output dir for merged files


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_readable_script(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing run script: {path}")
    if not os.access(path, os.R_OK):
        raise RuntimeError(f"Run script is not readable: {path}")


def split_events(total_events, chunk_size):
    chunks = []
    n_chunks = ceil(total_events / chunk_size)
    for idx in range(n_chunks):
        start = idx * chunk_size
        nev = min(chunk_size, total_events - start)
        chunks.append(nev)
    return chunks


def create_subjob_scripts():
    make_dir_if_not_exists(WORKDIR)
    make_dir_if_not_exists(f"{WORKDIR}/log")
    make_dir_if_not_exists(f"{WORKDIR}/out")
    make_dir_if_not_exists(f"{WORKDIR}/err")

    all_scripts = []
    chunks = split_events(TOTAL_EVENTS_PER_ANGLE, EVENTS_PER_JOB)

    for theta in ANGLES:
        theta_dir = f"{WORKDIR}/theta{theta}"
        make_dir_if_not_exists(theta_dir)

        for i, nev in enumerate(chunks, start=1):
            seed = SEED_BASE[theta] + i
            split_id = i
            sh_path = f"{theta_dir}/submit_theta{theta}_part{i}.sh"

            scr = "#!/usr/bin/env bash\n"
            scr += "set -euo pipefail\n\n"
            scr += f"cd {REPO}\n"
            scr += (
                f"bash {RUN_SCRIPT} {P_MIN} {P_MAX} {seed} {nev} {theta} {split_id}\n"
            )

            with open(sh_path, "w") as sh:
                sh.write(scr)

            os.chmod(sh_path, 0o755)
            all_scripts.append(sh_path)
            print(f"SUBMISSION SCRIPT CREATED: {sh_path}")

    return all_scripts


def create_condor_config(script_paths):
    cfg = "Universe = vanilla\n\n"
    cfg += f"initialdir = {REPO}\n"
    cfg += "executable = $(filename)\n\n"
    cfg += f"request_cpus = {REQUEST_CPUS}\n"
    cfg += f"request_memory = {REQUEST_MEMORY_MB}\n"
    cfg += 'accounting_group = cms.higgs\n'
    cfg += f"request_disk = {REQUEST_DISK_KB}\n"
    cfg += f"+RequestWalltime = {REQUEST_WALLTIME_HOURS}*3600\n\n"
    cfg += f"log = {WORKDIR}/log/gun_$(ClusterID).log\n"
    cfg += f"output = {WORKDIR}/out/gun_$(ClusterID)_$(ProcID).out\n"
    cfg += f"error = {WORKDIR}/err/gun_$(ClusterID)_$(ProcID).err\n\n"
    cfg += "queue filename from (\n"
    cfg += "\n".join([f"  {p}" for p in script_paths])
    cfg += "\n)\n"

    jdl_path = f"{WORKDIR}/job_submit.jdl"
    with open(jdl_path, "w") as sub:
        sub.write(cfg)

    print(f"CONDOR CONFIG CREATED: {jdl_path}")
    return jdl_path


def summarize(script_paths):
    jobs_per_angle = ceil(TOTAL_EVENTS_PER_ANGLE / EVENTS_PER_JOB)
    print("\nSUMMARY")
    print(f"  Momentum range: [{P_MIN}, {P_MAX}] GeV")
    print(f"  Angles: {ANGLES}")
    print(f"  Total events/angle: {TOTAL_EVENTS_PER_ANGLE}")
    print(f"  Events/job: {EVENTS_PER_JOB}")
    print(f"  Jobs/angle: {jobs_per_angle}")
    print(f"  Total jobs: {len(script_paths)}")
    print("  Expected outputs/job: sim + digi + fit")
    print(f"  Expected total output files: {len(script_paths) * 3}")


def submit_jobs(jdl_path, do_submit):
    if not do_submit:
        print("\nSubmit with:")
        print(f"  condor_submit {jdl_path}")
        return None

    proc = subprocess.run(
        ["condor_submit", jdl_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"condor_submit failed with exit code {proc.returncode}")

    text = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"submitted to cluster\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"cluster\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError("Could not parse cluster id from condor_submit output")

    return int(match.group(1))


def parse_merge_kinds(value):
    supported = {"sim", "digi", "fit"}
    kinds = [x.strip() for x in value.split(",") if x.strip()]
    if not kinds:
        raise ValueError("merge kinds cannot be empty")
    bad = [k for k in kinds if k not in supported]
    if bad:
        raise ValueError(f"unsupported merge kinds: {bad}, use sim,digi,fit")
    return kinds


def merge_outputs(kinds):
    if shutil.which("podio-merge-files") is None:
        raise RuntimeError(
            "podio-merge-files not found. Source Key4hep first, then rerun with --merge."
        )

    chunks = split_events(TOTAL_EVENTS_PER_ANGLE, EVENTS_PER_JOB)
    for theta in ANGLES:
        outdir = f"{OUTPUT_BASE}/deg{theta}"
        for kind in kinds:
            inputs = []
            missing = []

            for i, nev in enumerate(chunks, start=1):
                seed = SEED_BASE[theta] + i
                tag = (
                    f"{PARTICLE_TAG}_P{P_MIN}-{P_MAX}GeV_theta{theta}deg_seed{seed}_nev{nev}_part{i}"
                )
                path = f"{outdir}/{kind}_{tag}.root"
                if os.path.exists(path):
                    inputs.append(path)
                else:
                    missing.append(path)

            if missing:
                print(
                    f"SKIP MERGE {kind} theta={theta}: "
                    f"{len(missing)} missing chunk files"
                )
                continue

            merged = (
                f"{outdir}/{kind}_{PARTICLE_TAG}_P{P_MIN}-{P_MAX}GeV_"
                f"theta{theta}deg_nev{TOTAL_EVENTS_PER_ANGLE}_merged.root"
            )
            if os.path.exists(merged):
                os.remove(merged)

            cmd = ["podio-merge-files", "-o", merged] + inputs
            print(f"\nMerging {kind} theta={theta} -> {merged}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="")
            if proc.returncode != 0:
                raise RuntimeError(f"merge failed for {kind} theta={theta}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate split pion-gun submit scripts and JDL."
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs immediately after generating scripts and JDL.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge chunk outputs after production (or from existing files).",
    )
    parser.add_argument(
        "--merge-kinds",
        default=None,
        help="Comma-separated output types to merge: sim,digi,fit",
    )
    args = parser.parse_args()

    # If merge types are explicitly given, treat that as merge intent.
    if args.merge_kinds is not None and not args.merge:
        args.merge = True

    ensure_readable_script(RUN_SCRIPT)
    script_paths = create_subjob_scripts()
    jdl_path = create_condor_config(script_paths)
    summarize(script_paths)
    submit_jobs(jdl_path, args.submit)

    if args.merge:
        merge_kinds = parse_merge_kinds(args.merge_kinds or "sim,digi,fit")
        merge_outputs(merge_kinds)


if __name__ == "__main__":
    main()
