#!/usr/bin/env bash
set -euo pipefail

cd /work/omunkombwe/k4RecTracker
bash /work/omunkombwe/k4RecTracker/HTCondor/gun_fullsim_reco.sh 0.2 10.0 1003 1000 10 3
