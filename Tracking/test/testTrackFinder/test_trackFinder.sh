#!/bin/bash


MODEL_PATH=/work/omunkombwe/k4RecTracker/Tracking/test/inputFiles/SimpleGatrIDEAv3o1.onnx.md5


XML_FILE=$K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml
STEERING_FILE=SteeringFile_IDEA_o1_v03.py
TBETA=0.6
TD=0.3


curl -o $STEERING_FILE https://raw.githubusercontent.com/key4hep/k4geo/master/example/SteeringFile_IDEA_o1_v03.py


ddsim --steeringFile $STEERING_FILE \
      --compactFile  $XML_FILE \
      -G --gun.distribution uniform --gun.particle pi+ \
      --random.seed 42 \
      --numberOfEvents 1 \
      --outputFile out_sim_edm4hep.root


   
k4run runTestTrackFinder.py --inputFile out_sim_edm4hep.root --outputFile out_tracks.root --modelPath $MODEL_PATH --tbeta $TBETA --td $TD