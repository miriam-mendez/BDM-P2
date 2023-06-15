#! /bin/bash

echo "======================================================================================================================================================================"
echo "Running formatted zone and uploading files to HDFS."
python3 uploadToFormattedZone.py
echo "Finished formatted zone tasks."

echo "======================================================================================================================================================================"
echo "Getting data from formatted zone and upload it to exploitation zone..."
python3 uploadToExploitationZone.py
echo "Finished creating exploitation zone, KPI's were created."

echo "======================================================================================================================================================================"
echo "Training ML model..."
python3 MLmodel.py
echo "Model trained has complete and its measures are presented."