import os
import re
import glob
import ROOT
from podio.reading import get_reader
import edm4hep  # noqa: F401  (required for podio edm4hep types)
import math
import numpy as np
import matplotlib.pyplot as plt

# -- PHYSICS PARAMETERS --
B_FIELD = 2.0  # IDEA Central Solenoid in Tesla
BASE_DIR = "/ceph/omunkombwe/pi_Barrel"  # Base directory containing subdirs with merged ROOT files
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(
    r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root"
)
A_CONST = 2.99792458e-4  # 0.3 * 1e-3
ATIP_LOCATION = 1
PDG_ABS = 211
MIN_TRACKS = 1
P_MIN_GEV = 0.2
P_MAX_GEV = 10.0
P_BINS = 50
BARREL_ONLY = True
ENDCAP_ONLY = False
TRACK_COLLECTION_CANDIDATES = ["Fitted_tracks"]

# Region filtering: set one True for barrel/endcap; set both False or both True to keep all.
BARREL_DIGI_COLLECTIONS = [
    "DCH_DigiCollection",
    "VTXBDigis",
    "SiWrBDigis",
]
ENDCAP_DIGI_COLLECTIONS = [
    "VTXDDigis",
    "SiWrDDigis",
]

def get_value(obj, attr, getter=None, alt_getters=None):
    if getter and hasattr(obj, getter):
        return getattr(obj, getter)()
    if alt_getters:
        for alt in alt_getters:
            if hasattr(obj, alt):
                return getattr(obj, alt)()
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val() if callable(val) else val
    return None


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


def iter_tracker_hits(track):
    if hasattr(track, "trackerHits_size") and hasattr(track, "getTrackerHits"):
        n_hits = track.trackerHits_size()
        for i in range(n_hits):
            yield track.getTrackerHits(i)
        return
    hits = get_value(track, "trackerHits", "getTrackerHits")
    if hits is None:
        return
    for hit in hits:
        yield hit


def build_hit_oid_set(event, collection_names):
    oids = set()
    for name in collection_names:
        try:
            coll = event.get(name)
        except Exception:
            continue
        for hit in coll:
            oids.add(oid(hit))
    return oids


def is_barrel_track(track, barrel_oids, endcap_oids):
    hits = list(iter_tracker_hits(track))
    if not hits:
        return False

    barrel_count = 0
    endcap_count = 0
    for hit in hits:
        h_oid = oid(hit)
        if h_oid in barrel_oids:
            barrel_count += 1
        elif h_oid in endcap_oids:
            endcap_count += 1

    return endcap_count == 0 and barrel_count > 0


def is_endcap_track(track, barrel_oids, endcap_oids):
    hits = list(iter_tracker_hits(track))
    if not hits:
        return False

    barrel_count = 0
    endcap_count = 0
    for hit in hits:
        h_oid = oid(hit)
        if h_oid in barrel_oids:
            barrel_count += 1
        elif h_oid in endcap_oids:
            endcap_count += 1

    return barrel_count == 0 and endcap_count > 0


def get_vector3(vec):
    if vec is None:
        return None
    x = get_value(vec, "x", "getX")
    y = get_value(vec, "y", "getY")
    z = get_value(vec, "z", "getZ")
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def select_truth_momentum(mc_particles, pdg_abs):
    best = None
    best_p2 = -1.0

    for particle in mc_particles:
        gen = get_value(particle, "generatorStatus", "getGeneratorStatus")
        if gen is None or gen != 1:
            continue
        pdg = get_value(particle, "PDG", "getPDG", ["getPdg"])
        if pdg is None:
            pdg = get_value(particle, "pdg", None)
        if pdg_abs is not None and pdg is not None and abs(int(pdg)) != pdg_abs:
            continue

        mom = get_vector3(get_value(particle, "momentum", "getMomentum"))
        if mom is None:
            continue
        p2 = mom[0] ** 2 + mom[1] ** 2 + mom[2] ** 2
        if p2 > best_p2:
            best_p2 = p2
            best = mom

    if best is not None:
        return best

    for particle in mc_particles:
        mom = get_vector3(get_value(particle, "momentum", "getMomentum"))
        if mom is None:
            continue
        p2 = mom[0] ** 2 + mom[1] ** 2 + mom[2] ** 2
        if p2 > best_p2:
            best_p2 = p2
            best = mom

    return best


def choose_best_track(tracks):
    best_track = None
    best_val = None

    for track in tracks:
        chi2 = get_value(track, "chi2", "getChi2")
        ndf = get_value(track, "ndf", "getNdf")
        if chi2 is None or ndf is None:
            continue
        if chi2 <= 0 or ndf <= 0:
            continue

        states = get_value(track, "trackStates", "getTrackStates")
        if states is None or len(states) == 0:
            continue

        val = float(chi2) / float(ndf)
        if best_val is None or val < best_val:
            best_val = val
            best_track = track

    return best_track


def pick_track_state(track, atip_location):
    states = get_value(track, "trackStates", "getTrackStates")
    if states is None or len(states) == 0:
        return None

    chosen = None
    for state in states:
        loc = get_value(state, "location", "getLocation")
        if loc == atip_location:
            chosen = state
            break

    if chosen is None:
        chosen = states[0]

    omega = get_value(chosen, "omega", "getOmega")
    tan_lambda = get_value(chosen, "tanLambda", "getTanLambda")
    if omega is None or tan_lambda is None:
        return None

    return float(omega), float(tan_lambda)


def get_tracks_from_event(event):
    for coll_name in TRACK_COLLECTION_CANDIDATES:
        try:
            tracks = event.get(coll_name)
            if tracks is not None:
                return tracks
        except Exception:
            continue
    return []


def calculate_event_residuals(filename):
    """Processes one merged ROOT file and returns event-level arrays."""
    reader = get_reader(filename)
    events = reader.get("events")

    p_true_vals = []
    pt_true_vals = []
    residuals = []
    residuals_pt = []

    # Loop over events in the file
    for event in events:
        mc_particles = event.get("MCParticles")
        if len(mc_particles) == 0:
            continue

        truth_mom = select_truth_momentum(mc_particles, PDG_ABS)
        if truth_mom is None:
            continue

        px, py, pz = truth_mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        pt_true = math.hypot(px, py)

        if pt_true <= 0:
            continue
        if p_true <= 0:
            continue

        tracks = get_tracks_from_event(event)

        if len(tracks) == 0:
            continue
        if BARREL_ONLY or ENDCAP_ONLY:
            barrel_oids = build_hit_oid_set(event, BARREL_DIGI_COLLECTIONS)
            endcap_oids = build_hit_oid_set(event, ENDCAP_DIGI_COLLECTIONS)
            if BARREL_ONLY and not ENDCAP_ONLY:
                tracks = [t for t in tracks if is_barrel_track(t, barrel_oids, endcap_oids)]
            elif ENDCAP_ONLY and not BARREL_ONLY:
                tracks = [t for t in tracks if is_endcap_track(t, barrel_oids, endcap_oids)]
            # if both True or both False, keep all tracks
            if len(tracks) == 0:
                continue


        best_track = choose_best_track(tracks)
        if best_track is None:
            continue

        state = pick_track_state(best_track, ATIP_LOCATION)
        if state is None:
            continue

        omega_mm, tan_lambda = state
        if omega_mm == 0:

            continue

        pt_rec = A_CONST * B_FIELD / abs(omega_mm)
        p_rec = pt_rec * math.sqrt(1.0 + tan_lambda * tan_lambda)

        p_true_vals.append(p_true)
        pt_true_vals.append(pt_true)
        residuals.append((p_rec - p_true) / p_true)
        residuals_pt.append((pt_rec - pt_true) / pt_true)

    if len(residuals) < MIN_TRACKS or len(residuals_pt) < MIN_TRACKS:
        return None
    return (
        np.asarray(p_true_vals, dtype=float),
        np.asarray(pt_true_vals, dtype=float),
        np.asarray(residuals, dtype=float),
        np.asarray(residuals_pt, dtype=float),
    )


def binned_sigma(xvals, residuals, bin_edges, min_tracks):
    centers = []
    sigmas = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xvals >= lo) & (xvals < hi)
        if np.count_nonzero(mask) < min_tracks:
            continue
        centers.append(math.sqrt(lo * hi))
        sigmas.append(float(np.std(residuals[mask])))
    return centers, sigmas

# -- MAIN EXECUTION --
results_by_theta = {}

subdirs = sorted(glob.glob(os.path.join(BASE_DIR, SUBDIR_GLOB)))
if not subdirs:
    raise FileNotFoundError(f"No subdirectories found under {BASE_DIR} with pattern {SUBDIR_GLOB}")

for subdir in subdirs:
    fit_files = sorted(glob.glob(os.path.join(subdir, FIT_GLOB)))
    if not fit_files:
        print(f"Warning: no {FIT_GLOB} files in {subdir}")
        continue

    for file_path in fit_files:
        fname = os.path.basename(file_path)
        match = FILENAME_RE.match(fname)
        if not match:
            print(f"Warning: unrecognized filename format: {file_path}")
            continue

        theta = int(match.group("theta"))
        print(f"Analyzing merged file: {file_path} (theta={theta} deg)...")

        res = calculate_event_residuals(file_path)
        if res is None:
            continue
        p_true, pt_true, residuals, residuals_pt = res
        theta_store = results_by_theta.setdefault(
            theta,
            {"p_true": [], "pt_true": [], "residuals": [], "residuals_pt": []},
        )
        theta_store["p_true"].append(p_true)
        theta_store["pt_true"].append(pt_true)
        theta_store["residuals"].append(residuals)
        theta_store["residuals_pt"].append(residuals_pt)

bin_edges = np.logspace(np.log10(P_MIN_GEV), np.log10(P_MAX_GEV), P_BINS + 1)
final_results = {}
final_results_pt = {}
momenta_by_theta = {}
momenta_by_theta_pt = {}

for theta, vals in results_by_theta.items():
    p_true = np.concatenate(vals["p_true"]) if vals["p_true"] else np.asarray([], dtype=float)
    pt_true = np.concatenate(vals["pt_true"]) if vals["pt_true"] else np.asarray([], dtype=float)
    residuals = np.concatenate(vals["residuals"]) if vals["residuals"] else np.asarray([], dtype=float)
    residuals_pt = np.concatenate(vals["residuals_pt"]) if vals["residuals_pt"] else np.asarray([], dtype=float)

    if p_true.size == 0:
        continue

    p_centers, sigmas = binned_sigma(p_true, residuals, bin_edges, MIN_TRACKS)

    pt_scale = math.sin(math.radians(theta))
    if pt_scale <= 0.0:
        pt_centers, sigmas_pt = [], []
    else:
        pt_edges = bin_edges * pt_scale
        pt_centers, sigmas_pt = binned_sigma(pt_true, residuals_pt, pt_edges, MIN_TRACKS)

    momenta_by_theta[theta] = p_centers
    final_results[theta] = sigmas
    momenta_by_theta_pt[theta] = pt_centers
    final_results_pt[theta] = sigmas_pt

# -- PLOTTING --
plt.figure(figsize=(10, 7))
for theta in sorted(final_results.keys()):
    plt.plot(momenta_by_theta[theta], final_results[theta], 'o--', label=f'Theta = {theta}°')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Generated Momentum $p$ [GeV/c]', fontsize=12)
plt.ylabel(r'Momentum Resolution $\sigma_p / p$', fontsize=12)
plt.title('Pion Momentum Resolution', fontsize=14)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.savefig("pi_Barrel_Momentum_Resolution.png", dpi=300)
plt.show()

# -- PT PLOTTING --
plt.figure(figsize=(10, 7))
for theta in sorted(final_results_pt.keys()):
    if not final_results_pt[theta]:
        continue
    plt.plot(momenta_by_theta_pt[theta], final_results_pt[theta], "o--", label=f"Theta = {theta}°")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("Generated Transverse Momentum $p_T$ [GeV/c]", fontsize=12)
plt.ylabel(r"Transverse Momentum Resolution $\sigma_{p_T} / p_T$", fontsize=12)
plt.title("Pion $p_T$ Resolution", fontsize=14)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.savefig("pi_Barrel_Pt_Resolution.png", dpi=300)
plt.show()
