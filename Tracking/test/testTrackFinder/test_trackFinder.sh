#!/bin/bash


MODEL_PATH=/work/omunkombwe/k4RecTracker/build/Tracking/test/inputFiles/SimpleGatrIDEAv3o1.onnx


XML_FILE=$K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml
STEERING_FILE=/work/omunkombwe/k4RecTracker/my_py/SteeringFile_IDEA_o1_v03.py
TBETA=0.6
TD=0.3


#curl -o $STEERING_FILE https://raw.githubusercontent.com/key4hep/k4geo/master/example/SteeringFile_IDEA_o1_v03.py


ddsim \
  --steeringFile $STEERING_FILE \
  --compactFile $XML_FILE \
  -G \
  --gun.distribution uniform \
  --gun.momentumMin "0.5*GeV" \
  --gun.momentumMax "5*GeV" \
  --gun.particle mu- \
  --gun.thetaMin "60*degree" \
  --gun.thetaMax "60*degree" \
  --numberOfEvents 100 \
  --random.enableEventSeed \
  --random.seed 10 \
  --outputFile /ceph/omunkombwe/out_sim_idea_o1_v03_60.root


k4run /work/omunkombwe/k4RecTracker/Tracking/test/testTrackFinder/runTestTrackFinder.py --inputFile /ceph/omunkombwe/out_sim_idea_o1_v03_60.root --outputFile /ceph/omunkombwe/out_tracks1_60.root --modelPath $MODEL_PATH --tbeta $TBETA --td $TD
k4run /work/omunkombwe/k4RecTracker/Tracking/test/testTrackFitter/runTestTrackFitter.py --inputFile /ceph/omunkombwe/out_tracks1_60.root --outputFile /ceph/omunkombwe/output_tracks1_new_60.root