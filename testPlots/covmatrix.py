#!/usr/bin/env python3
from podio.reading import get_reader
import edm4hep

path = "/ceph/omunkombwe/fit_pi.root"
max_events = 5

# covMatrix order: (D0, phi, omega, Z0, tanLambda, time)
PARAMS = ["D0", "phi", "omega", "Z0", "tanLambda", "time"]


def cov_idx(i, j):
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j


reader = get_reader(path)
events = reader.get("events")

for evt, event in enumerate(events):
    if evt >= max_events:
        break
    tracks = event.get("Fitted_tracks")

    for it, trk in enumerate(tracks):
        for ts in trk.getTrackStates():
            if ts.location == edm4hep.TrackState.AtIP:
                break

        cov_matrix = [
            [float(ts.covMatrix[cov_idx(i, j)]) for j in range(len(PARAMS))]
            for i in range(len(PARAMS))
        ]

        print("evt", evt, "track", it)
        print("covMatrix (rows: D0, phi, omega, Z0, tanLambda, time)")
        for name, row in zip(PARAMS, cov_matrix):
            print(name, row)
        print("-")
