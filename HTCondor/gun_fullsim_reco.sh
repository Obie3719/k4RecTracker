#!/usr/bin/env bash
set -euo pipefail

P_MIN="${1:-3}"
P_MAX="${2:-3}"
SEED="${3:-42}"
NEV="${4:-5000}"
THETA_DEG=60

REPO=/work/omunkombwe/k4RecTracker
MODEL_PATH=$REPO/build/Tracking/test/inputFiles/SimpleGatrIDEAv3o1.onnx
STEERING_FILE=$REPO/my_py/SteeringFile_IDEA_o1_v03.py

OUTDIR=/ceph/omunkombwe/idea_o1_v03_gun
mkdir -p "$OUTDIR"
TAG="mu_P${P_MIN}-${P_MAX}GeV_theta${THETA_DEG}deg_seed${SEED}_nev${NEV}"
SIM_OUTPUT="${OUTDIR}/sim_${TAG}.root"
FINDER_OUTPUT="${OUTDIR}/digi_${TAG}.root"
FITTER_OUTPUT="${OUTDIR}/fit_${TAG}.root"

# ---- setup FIRST ----
cd "$REPO"
set +u
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2025-12-21
k4_local_repo
set -u

# now K4GEO exists
: "${K4GEO:?K4GEO is not set—Key4hep setup failed}"
XML_FILE=$K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml

# ---- run ----
ddsim \
  --steeringFile "$STEERING_FILE" \
  --compactFile "$XML_FILE" \
  -G \
  --gun.distribution uniform \
  --gun.momentumMin "${P_MIN}*GeV" \
  --gun.momentumMax "${P_MAX}*GeV" \
  --gun.particle mu- \
  --gun.thetaMin "${THETA_DEG}*degree" \
  --gun.thetaMax "${THETA_DEG}*degree" \
  --numberOfEvents "$NEV" \
  --random.enableEventSeed \
  --random.seed "$SEED" \
  --outputFile "$SIM_OUTPUT"

k4run "$REPO/Tracking/test/testTrackFinder/runTestTrackFinder.py" \
  --inputFile "$SIM_OUTPUT" \
  --outputFile "$FINDER_OUTPUT" \
  --modelPath "$MODEL_PATH" --tbeta 0.6 --td 0.3

k4run "$REPO/Tracking/test/testTrackFitter/runTestTrackFitter.py" \
  --inputFile "$FINDER_OUTPUT" \
  --outputFile "$FITTER_OUTPUT"
