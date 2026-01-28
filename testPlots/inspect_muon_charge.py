#!/usr/bin/env python3
import argparse
from collections import Counter

from podio.reading import get_reader


def mc_charge_from_hits(track):
    """Return the dominant MC particle charge linked via tracker hits, or None if unavailable."""
    charges = []
    try:
        nhits = track.trackerHits_size()
    except Exception:
        nhits = 0
    for ihit in range(nhits):
        try:
            hit = track.getTrackerHits(ihit)
        except Exception:
            continue
        # Hit -> MC particle relation
        mc = None
        for getter in ("getParticle", "getMCParticle", "particle", "MCParticle"):
            try:
                attr = getattr(hit, getter)
                mc = attr() if callable(attr) else attr
                if mc is not None:
                    break
            except Exception:
                pass
        if mc is None:
            continue
        for getter in ("getCharge", "charge"):
            try:
                attr = getattr(mc, getter)
                val = attr() if callable(attr) else attr
                if val is not None:
                    charges.append(val)
                    break
            except Exception:
                pass
    if not charges:
        return None
    # Use the majority sign among associated MC particles
    plus = sum(1 for c in charges if c > 0)
    minus = sum(1 for c in charges if c < 0)
    if plus == minus == 0:
        return 0
    return 1 if plus >= minus else -1


def infer_charge(track, state):
    # Prefer an explicit charge getter if present
    for getter in ("getCharge", "charge"):
        try:
            attr = getattr(track, getter)
            val = attr() if callable(attr) else attr
            if val is not None:
                return 1 if val > 0 else -1 if val < 0 else 0
        except Exception:
            pass

    # Some workflows encode the sign in the track type
    try:
        t = track.getType()
        if t != 0:
            return 1 if t > 0 else -1
    except Exception:
        pass

    # Fall back to the track state curvature sign (omega)
    if state is not None and getattr(state, "omega", 0) != 0:
        return 1 if state.omega > 0 else -1
    return 0


def scan_muon_charges(path, max_events, show_omega=False, prefer_truth=False):
    reader = get_reader(path)
    events = reader.get("events")

    counts = Counter()
    seen_events = 0

    for ievt, event in enumerate(events):
        if ievt >= max_events:
            break
        seen_events += 1
        tracks = event.get("Fitted_tracks_muon")

        entries = []
        for trk in tracks:
            state = trk.getTrackStates(0) if trk.trackStates_size() else None
            omega = state.omega if state is not None else 0.0
            q_truth = mc_charge_from_hits(trk) if prefer_truth else None
            q = q_truth if q_truth is not None else infer_charge(trk, state)
            counts[q] += 1
            entries.append((q, omega, q_truth) if show_omega else (q, q_truth))

        if show_omega:
            pretty = ", ".join(
                f"q={q}, omega={omega:.4g}" + (f", q_truth={qt}" if qt is not None else "")
                for q, omega, qt in entries
            )
        else:
            pretty = ", ".join(
                str(q) if qt is None else f"{q} (truth {qt})" for q, qt in entries
            )
        print(f"Event {ievt}: {pretty}")

    print(f"Totals over {seen_events} events -> q+: {counts[1]}, q-: {counts[-1]}, q0: {counts[0]}")


def main():
    parser = argparse.ArgumentParser(description="Inspect muon track charges in a PODIO EDM4hep file.")
    parser.add_argument(
        "--input",
        default="/ceph/omunkombwe/fitter_idea_mumu50k.root",
        help="Input ROOT file path (PODIO EDM4hep).",
    )
    parser.add_argument(
        "-n",
        "--events",
        type=int,
        default=10,
        help="Number of events to scan.",
    )
    parser.add_argument(
        "--show-omega",
        action="store_true",
        help="Also print the omega curvature value used to infer charge.",
    )
    parser.add_argument(
        "--prefer-truth",
        action="store_true",
        help="Use MC particle charge via tracker-hit associations when available (falls back to omega sign otherwise).",
    )
    args = parser.parse_args()

    scan_muon_charges(args.input, args.events, show_omega=args.show_omega, prefer_truth=args.prefer_truth)


if __name__ == "__main__":
    main()
