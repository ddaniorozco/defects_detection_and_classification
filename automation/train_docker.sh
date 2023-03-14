#!/usr/bin/env bash

# Check GPU is working:
# tf1
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf2
# tf.test.is_gpu_available()
# or
# tf.config.list_physical_devices('GPU')

if [[ -z "${WORKSPACE}" ]]; then
  export WORKSPACE="$(pwd)/workspace"
fi

# Preparing environment variables so the docker image would have access to aws while running tests
echo "Preparing environment variables for the tensorflow docker image..."
export ENV_LIST_FILE_NAME="${WORKSPACE}/env.list"
echo " -> env | grep AWS > env.list"
env | grep AWS > "${ENV_LIST_FILE_NAME}"
echo "REMOTE_PIPELINE_CONFIG_PATH=${REMOTE_PIPELINE_CONFIG_PATH}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "RUN_DISPLAY_URL=${RUN_DISPLAY_URL}"
echo "BUILD_ID=${BUILD_ID}"
echo "REMOTE_PIPELINE_CONFIG_PATH=${REMOTE_PIPELINE_CONFIG_PATH}" >> "${ENV_LIST_FILE_NAME}"
echo "MODEL_DIR=${MODEL_DIR}" >> "${ENV_LIST_FILE_NAME}"
echo "RUN_DISPLAY_URL=${RUN_DISPLAY_URL}" >> "${ENV_LIST_FILE_NAME}"
echo "BUILD_ID=${BUILD_ID}" >> "${ENV_LIST_FILE_NAME}"
#echo "env.list content:"
#cat "${ENV_LIST_FILE_NAME}"

echo "docker login..."
export REPOSITORY="hawkeyetraining.azurecr.io"
export REPO_USERNAME="hawkeyeTraining"
export DOCKER_IMAGE_NAME="od_hawkeye_training"
docker login --username "${REPO_USERNAME}" --password "${AZURE_DOCKER_TRAINING_REPOSITORY_PASSWORD}" "${REPOSITORY}"

# Pulling docker image
REPOSITORY_TAG_FOR_LATEST="${REPOSITORY}/${DOCKER_IMAGE_NAME}:latest"
echo "Pulling hawkeye training docker image from ${REPOSITORY_TAG_FOR_LATEST}..."
docker pull "${REPOSITORY_TAG_FOR_LATEST}"

export DOCKER_CONT_NAME="odht"

echo "Checking if nvidia gpu exists"
lspci
total_nvidia_gpu=$(lspci | grep NVIDIA | wc -l)
if ((total_nvidia_gpu > 0)); then
  echo "${total_nvidia_gpu} gpu was found"
  export DOCKER_GPU_ARG=" --gpus all"
else
  echo "gpu wasn't found"
  export DOCKER_GPU_ARG=""
fi

# Print tensorboard link

echo "Running training inside docker container..."
echo " -> docker run --name=${DOCKER_CONT_NAME}${DOCKER_GPU_ARG} --rm --env-file ${ENV_LIST_FILE_NAME} -v ${WORKSPACE}:/workspace -t ${REPOSITORY_TAG_FOR_LATEST} /bin/bash /workspace/automation/train.sh"
docker run --name="${DOCKER_CONT_NAME}"${DOCKER_GPU_ARG} --rm --env-file "${ENV_LIST_FILE_NAME}" -p 6006:6006 -v "${WORKSPACE}":/workspace -t "${REPOSITORY_TAG_FOR_LATEST}" /bin/bash /workspace/automation/train.sh
