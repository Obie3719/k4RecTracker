#!/usr/bin/env python3
import argparse
import math

import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


DEFAULT_INPUT = "/ceph/omunkombwe/fitter_idea_mumu50k.root"


def _diag_index(i):
    # Lower-triangular row-wise storage.
    return i * (i + 1) // 2 + i


def _print_quantiles(name, data):
    print(f"{name}: count={data.size}")
    if data.size == 0:
        return
    for q in (0.5, 0.9, 0.95, 0.99, 0.999):
        print(f"  q{int(q * 1000):4d}: {np.quantile(data, q):.6g}")
    print(f"  max: {data.max():.6g}")


def _plot_hist(ax, data, title, xlabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tracks")
    if data.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return
    hi = np.quantile(data, 0.995)
    if not np.isfinite(hi) or hi <= 0:
        hi = data.max()
    bins = np.linspace(0.0, hi, 60) if hi > 0 else 50
    ax.hist(
        data,
        bins=bins,
        histtype="stepfilled",
        alpha=0.75,
        color="C0",
        edgecolor="black",
    )


def extract_sigmas(path, collection, bfield_tesla, max_events=None):
    with uproot.open(path) as f:
        event_keys = [k for k in f.keys() if k.startswith("events;")]
        if not event_keys:
            raise RuntimeError("No events tree found in file.")
        events_key = max(event_keys, key=lambda name: int(name.split(";")[1]))
        tree = f[events_key]

        cov_base = f"_{collection}_trackStates/_{collection}_trackStates.covMatrix.values"
        cov_branch = None
        for size in (21, 15):
            candidate = f"{cov_base}[{size}]"
            if candidate in tree.keys():
                cov_branch = candidate
                break
        if cov_branch is None:
            raise RuntimeError(f"covMatrix branch not found for {collection}.")

        begin_branch = f"{collection}/{collection}.trackStates_begin"
        end_branch = f"{collection}/{collection}.trackStates_end"
        for branch in (begin_branch, end_branch):
            if branch not in tree.keys():
                raise RuntimeError(f"Missing branch: {branch}")

    sig_d0 = []
    sig_z0 = []
    sig_1pt = []
    neg_cov = {"d0": 0, "z0": 0, "omega": 0}
    used_tracks = 0

    branches = [cov_branch, begin_branch, end_branch]
    for arrays in uproot.iterate(
        f"{path}:{events_key}",
        branches,
        library="ak",
        entry_stop=max_events,
        step_size="50 MB",
    ):
        cov = arrays[cov_branch]
        begin = arrays[begin_branch]
        end = arrays[end_branch]

        track_counts = end - begin
        mask = track_counts > 0
        begin_first = begin[mask]
        cov_first = cov[begin_first]

        flat_cov = ak.to_numpy(ak.flatten(cov_first, axis=1))
        if flat_cov.size == 0:
            continue

        used_tracks += flat_cov.shape[0]

        d0_diag = flat_cov[:, _diag_index(0)]
        omega_diag = flat_cov[:, _diag_index(2)]
        z0_diag = flat_cov[:, _diag_index(3)]

        for name, diag in (("d0", d0_diag), ("omega", omega_diag), ("z0", z0_diag)):
            neg_cov[name] += int(np.count_nonzero(diag < 0))

        d0_diag = d0_diag[d0_diag >= 0]
        z0_diag = z0_diag[z0_diag >= 0]
        omega_diag = omega_diag[omega_diag >= 0]

        if d0_diag.size:
            sig_d0.append(np.sqrt(d0_diag))
        if z0_diag.size:
            sig_z0.append(np.sqrt(z0_diag))
        if omega_diag.size:
            sigma_omega = np.sqrt(omega_diag)
            kappa = 0.299792458 * bfield_tesla
            sig_1pt.append((1000.0 / kappa) * sigma_omega)

    sig_d0 = np.concatenate(sig_d0) if sig_d0 else np.array([])
    sig_z0 = np.concatenate(sig_z0) if sig_z0 else np.array([])
    sig_1pt = np.concatenate(sig_1pt) if sig_1pt else np.array([])

    return sig_d0, sig_z0, sig_1pt, used_tracks, neg_cov


def main():
    parser = argparse.ArgumentParser(description="TrackState uncertainty summary.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="EDM4hep ROOT file.")
    parser.add_argument("--collection", default="Fitted_tracks_muon", help="Track collection.")
    parser.add_argument("--bfield", type=float, default=2.0, help="B-field in Tesla.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event limit.")
    parser.add_argument("--out", default="trackstate_sigmas.png", help="Output plot file.")
    args = parser.parse_args()

    sig_d0, sig_z0, sig_1pt, used_tracks, neg_cov = extract_sigmas(
        args.input, args.collection, args.bfield, args.max_events
    )

    print(f"collection: {args.collection}")
    print(f"tracks used (first state): {used_tracks}")
    print(f"negative cov diag counts: {neg_cov}")
    _print_quantiles("sigma(d0)", sig_d0)
    _print_quantiles("sigma(z0)", sig_z0)
    _print_quantiles("sigma(1/pT) [1/GeV]", sig_1pt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    _plot_hist(axes[0], sig_d0, "sigma(d0)", "sigma(d0) [mm]")
    _plot_hist(axes[1], sig_z0, "sigma(z0)", "sigma(z0) [mm]")
    _plot_hist(axes[2], sig_1pt, "sigma(1/pT)", "sigma(1/pT) [1/GeV]")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
