#!/usr/bin/env bash

export PYTHONUNBUFFERED="TRUE"

echo "TRAIN_IMAGE_DIR=${TRAIN_IMAGE_DIR}"
echo "VAL_IMAGE_DIR=${VAL_IMAGE_DIR}"
echo "TEST_IMAGE_DIR=${TEST_IMAGE_DIR}"
echo "TRAIN_ANNOTATIONS_FILE=${TRAIN_ANNOTATIONS_FILE}"
echo "VAL_ANNOTATIONS_FILE=${VAL_ANNOTATIONS_FILE}"
echo "TESTDEV_ANNOTATIONS_FILE=${TESTDEV_ANNOTATIONS_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

cp "/workspace/automation/create_coco_tf_records_fix.py" /home/tensorflow/models/research/object_detection/dataset_tools/

cd /home/tensorflow/models/research/tf_records/
mkdir output

aws s3 cp "${TRAIN_IMAGE_DIR}" /home/tensorflow/models/research/tf_records/train_images/ --recursive
aws s3 cp "${VAL_IMAGE_DIR}" /home/tensorflow/models/research/tf_records/val_images/ --recursive
aws s3 cp "${TEST_IMAGE_DIR}" /home/tensorflow/models/research/tf_records/test_images/ --recursive
aws s3 cp "${TRAIN_ANNOTATIONS_FILE}" /home/tensorflow/models/research/tf_records/train.json
aws s3 cp "${VAL_ANNOTATIONS_FILE}" /home/tensorflow/models/research/tf_records/val.json
aws s3 cp "${TESTDEV_ANNOTATIONS_FILE}" /home/tensorflow/models/research/tf_records/test.json

echo "Creating TF records..."
python3 /home/tensorflow/models/research/object_detection/dataset_tools/create_coco_tf_records_fix.py \
  --logtostderr \
  --train_image_dir="/home/tensorflow/models/research/train_images/" \
  --val_image_dir="/home/tensorflow/models/research/val_images/" \
  --test_image_dir="/home/tensorflow/models/research/test_images/" \
  --train_annotations_file="/home/tensorflow/models/research/tf_records/train.json" \
  --val_annotations_file="/home/tensorflow/models/research/tf_records/val.json" \
  --testdev_annotations_file="/home/tensorflow/models/research/tf_records/test.json" \
  --output_dir="output"

aws s3 cp output "${OUTPUT_DIR}" --recursive

echo "TF records finished!"