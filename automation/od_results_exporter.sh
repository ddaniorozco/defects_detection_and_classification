#!/usr/bin/env bash

export PYTHONUNBUFFERED="TRUE"

#export "/workspace/automation/od_results_exporter.py"
#export "/workspace/automation/tf_object_detector.py"

echo "FROZEN_GRAPH_PATH=${FROZEN_GRAPH_PATH}"
echo "LABELS_PATH=${LABELS_PATH}"
echo "MIN_THRESH_PERCENT=${MIN_THRESH_PERCENT}"
echo "IMAGES_PATH=${IMAGES_PATH}"
echo "RESULTS_JSON=${RESULTS_JSON}"
echo "RESULTS_IMAGES=${RESULTS_IMAGES}"

cp "/workspace/automation/od_results_exporter.py" /home/tensorflow/models/research/
cp "/workspace/automation/tf_object_detector.py" /home/tensorflow/models/research/

cd /home/tensorflow/models/research
ls

aws s3 cp "${FROZEN_GRAPH_PATH}" /home/tensorflow/models/research/frozen_inference_graph.pb
aws s3 cp "${LABELS_PATH}" /home/tensorflow/models/research/labelmap_13classes_november.pbtxt
aws s3 cp "${IMAGES_PATH}" /home/tensorflow/models/research/images/ --recursive
mkdir results_json
mkdir results_images

echo "Exporting Object Detection results..."
python3 /home/tensorflow/models/research/od_results_exporter.py \
    --frozen_graph_path="/home/tensorflow/models/research/frozen_inference_graph.pb" \
    --labels_path="/home/tensorflow/models/research/labelmap_13classes_november.pbtxt" \
    --min_thresh_percent="${MIN_THRESH_PERCENT}" \
    --images_path="/home/tensorflow/models/research/images/*" \
    --results_json="/home/tensorflow/models/research/results_json/" \
    --results_images="/home/tensorflow/models/research/results_images/"

aws s3 cp results_json "${RESULTS_JSON}" --recursive
aws s3 cp results_images "${RESULTS_IMAGES}" --recursive

rm -r results_json
rm -r results_images

echo "Object detection results exporter finished!"
