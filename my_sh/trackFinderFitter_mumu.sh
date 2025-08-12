#!/bin/bash

MODEL_PATH=/work/omunkombwe/k4RecTracker/build/Tracking/test/inputFiles/SimpleGatrIDEAv3o1.onnx

XML_FILE=$K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml
STEERING_FILE=Tracking/test/testTrackFinder/SteeringFile_IDEA_o1_v03.py
TBETA=0.6
TD=0.3

HEPMC_INPUT=/work/omunkombwe/k4RecTracker/my_HEPMC/ee_z_mumu100.hepmc

curl -o $STEERING_FILE https://raw.githubusercontent.com/key4hep/k4geo/master/example/SteeringFile_IDEA_o1_v03.py

ddsim --steeringFile $STEERING_FILE \
      --compactFile  $XML_FILE \
      --inputFile $HEPMC_INPUT \
      --random.seed 42 \
      --numberOfEvents 100 \
      --outputFile output_rootFiles/ee_z_mumu0.root 
    
k4run Tracking/test/testTrackFinder/runTestTrackFinder.py --inputFile output_rootFiles/ee_z_mumu0.root --outputFile output_rootFiles/ee_z_mumu1.root --modelPath $MODEL_PATH --tbeta $TBETA --td $TD

k4run Tracking/test/testTrackFitter/runTestTrackFitter.py --inputFile output_rootFiles/ee_z_mumu1.root --outputFile output_rootFiles/ee_z_mumu2.root
