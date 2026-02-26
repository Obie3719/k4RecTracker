#!/usr/bin/env bash
set -euo pipefail

cd /work/omunkombwe/k4RecTracker
bash /work/omunkombwe/k4RecTracker/HTCondor/gun_fullsim_reco.sh 0.2 10.0 3084 1000 30 84
