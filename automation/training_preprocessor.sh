#!/usr/bin/env bash

export PYTHONUNBUFFERED="TRUE"



echo "SOURCE_DIR=${SOURCE_DIR}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "VAL_DIR=${VAL_DIR}"
echo "TEST_PERCENTAGE=${TEST_PERCENTAGE}"
echo "VAL_PERCENTAGE=${VAL_PERCENTAGE}"

cd /home/tensorflow/models/research
mkdir training_preprocess

cp "/workspace/automation/labelme2coco.py" /home/tensorflow/models/research/training_preprocess
cp "/workspace/automation/training_preprocessor.py" /home/tensorflow/models/research/training_preprocess

cd training_preprocess
ls

aws s3 cp "${SOURCE_DIR}" /home/tensorflow/models/research/training_preprocess/source_dir/ --recursive
mkdir train_dir
mkdir test_dir
mkdir val_dir

pip3 install labelme==5.0.5
pip3 install tqdm

echo "Training preprocess..."
python3 /home/tensorflow/models/research/training_preprocess/training_preprocessor.py \
    --source_folder="/home/tensorflow/models/research/training_preprocess/source_dir/" \
    --train_folder="/home/tensorflow/models/research/training_preprocess/train_dir/" \
    --test_folder="/home/tensorflow/models/research/training_preprocess/test_dir/" \
    --val_folder="/home/tensorflow/models/research/training_preprocess/val_dir/" \
    --test_percentage="${TEST_PERCENTAGE}" \
    --val_percentage="${VAL_PERCENTAGE}"

aws s3 cp train_dir "${TRAIN_DIR}" --recursive
aws s3 cp test_dir "${TEST_DIR}" --recursive
aws s3 cp val_dir "${VAL_DIR}" --recursive

cd /home/tensorflow/models/research
rm -r training_preprocess

echo "Training preprocess finished!"