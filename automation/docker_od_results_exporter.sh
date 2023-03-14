#!/usr/bin/env bash

if [[ -z "${WORKSPACE}" ]]; then
  export WORKSPACE="$(pwd)/workspace"
fi

echo "Preparing environment variables for the tensorflow docker image..."
export ENV_LIST_FILE_NAME="${WORKSPACE}/env.list"
echo " -> env | grep AWS > env.list"
env | grep AWS > "${ENV_LIST_FILE_NAME}"

echo "FROZEN_GRAPH_PATH=${FROZEN_GRAPH_PATH}" >> "${ENV_LIST_FILE_NAME}"
echo "LABELS_PATH=${LABELS_PATH}" >> "${ENV_LIST_FILE_NAME}"
# echo "NUM_CLASSES=${NUM_CLASSES}" >> "${ENV_LIST_FILE_NAME}"
echo "MIN_THRESH_PERCENT=${MIN_THRESH_PERCENT}" >> "${ENV_LIST_FILE_NAME}"
echo "IMAGES_PATH=${IMAGES_PATH}" >> "${ENV_LIST_FILE_NAME}"
echo "RESULTS_JSON=${RESULTS_JSON}" >> "${ENV_LIST_FILE_NAME}"
echo "RESULTS_IMAGES=${RESULTS_IMAGES}" >> "${ENV_LIST_FILE_NAME}"
echo "RUN_DISPLAY_URL=${RUN_DISPLAY_URL}" >> "${ENV_LIST_FILE_NAME}"
echo "BUILD_ID=${BUILD_ID}" >> "${ENV_LIST_FILE_NAME}"
echo "WORKSPACE=${WORKSPACE}" >> "${ENV_LIST_FILE_NAME}"

echo "docker login..."
export REPOSITORY="hawkeyetraining.azurecr.io"
export REPO_USERNAME="hawkeyeTraining"
export DOCKER_IMAGE_NAME="od_hawkeye_training"
docker login --username "${REPO_USERNAME}" --password "${AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD}" "${REPOSITORY}"

REPOSITORY_TAG_FOR_LATEST="${REPOSITORY}/${DOCKER_IMAGE_NAME}:latest"
echo "Pulling hawkeye training docker image from ${REPOSITORY_TAG_FOR_LATEST}..."
docker pull "${REPOSITORY_TAG_FOR_LATEST}"

export DOCKER_CONT_NAME="od_results_exporter"

echo "Running training inside docker container..."
echo " -> docker run --name=${DOCKER_CONT_NAME} --rm --env-file ${ENV_LIST_FILE_NAME} -v ${WORKSPACE}:/workspace -t ${REPOSITORY_TAG_FOR_LATEST} /bin/bash /workspace/automation/od_results_exporter.sh"
docker run --name="${DOCKER_CONT_NAME}" --rm --env-file "${ENV_LIST_FILE_NAME}" -v "${WORKSPACE}":/workspace -t "${REPOSITORY_TAG_FOR_LATEST}" /bin/bash /workspace/automation/od_results_exporter.sh
