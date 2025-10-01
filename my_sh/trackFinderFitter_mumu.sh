#!/bin/bash
#--random.seed 42 \

MODEL_PATH=/work/omunkombwe/k4RecTracker/build/Tracking/test/inputFiles/SimpleGatrIDEAv3o1.onnx

XML_FILE=$K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml
STEERING_FILE=/work/omunkombwe/k4RecTracker/my_py/SteeringFile_IDEA_o1_v03.py
TBETA=0.6
TD=0.3

HEPMC_INPUT=/work/omunkombwe/k4RecTracker/my_HEPMC/ee_mumu10k.hepmc
SIM_EVENTS=${SIM_EVENTS:-10000}
RANDOM_SEED=${RANDOM_SEED:-42}

SIM_OUTPUT=/ceph/omunkombwe/ee_z_mumu_sim.root
FINDER_OUTPUT=/ceph/omunkombwe/ee_z_mumu_finder.root
FITTER_OUTPUT=/ceph/omunkombwe/ee_z_mumu_fitter.root

#curl -o $STEERING_FILE https://raw.githubusercontent.com/key4hep/k4geo/master/example/SteeringFile_IDEA_o1_v03.py

ddsim --steeringFile $STEERING_FILE \
      --compactFile  $XML_FILE \
      --inputFile $HEPMC_INPUT \
      --numberOfEvents $SIM_EVENTS \
      --random.enableEventSeed \
      --random.seed $RANDOM_SEED \
      --runType batch \
      --outputFile $SIM_OUTPUT

k4run /work/omunkombwe/k4RecTracker/Tracking/test/testTrackFinder/runTestTrackFinder.py --inputFile $SIM_OUTPUT --outputFile $FINDER_OUTPUT --modelPath $MODEL_PATH --tbeta $TBETA --td $TD
k4run /work/omunkombwe/k4RecTracker/Tracking/test/testTrackFitter/runTestTrackFitter.py --inputFile $FINDER_OUTPUT --outputFile $FITTER_OUTPUT
