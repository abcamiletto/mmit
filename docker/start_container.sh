# Define Variables
DOCKER_IMAGE_NAME="mmit"
GPUS=""
RAM="8g"
CONTAINER_NAME="mmit_container"
REPO_DIR="$(dirname $(dirname $(realpath $0)))"
COMMAND="pip install --no-deps -e ."

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gpus) gpus="$2"; shift ;;
        -r|--ram) ram="$2"; shift ;;
        *) echo "Usage: $0 [-g|--gpus <gpus>] [-r|--ram <ram>]"; exit 1 ;;
    esac
    shift
done

# Check if gpu are available in the current machine
if [ "$(nvidia-smi -L | wc -l)" -eq 0 ]; then
    echo "No GPU available in the current machine, using CPU only"
    GPUS=""
fi

# Check if nvidia-docker is installed
if ! command -v nvidia-docker &> /dev/null; then
    echo "nvidia-docker could not be found, running with docker"
fi

# Prepare the docker arguments

if [ -z "$GPUS" ]; then
    GPUS_ARG=""
else
    GPUS_ARG="--gpus $GPUS"
fi
RAM_ARG="--memory $RAM"
VOLUMES_ARG="-v $REPO_DIR:/workspace"
NAME_ARG="--name $CONTAINER_NAME"

# Start the container
docker run -it --rm $RAM_ARG $GPUS_ARG $VOLUMES_ARG $NAME_ARG \
$DOCKER_IMAGE_NAME \
/bin/bash -c "$COMMAND && /bin/bash"
