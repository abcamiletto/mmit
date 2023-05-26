# Get the directory of this script
SCRIPT_DIR="$(dirname "$0")"

# Build the image
docker build -t mmit $SCRIPT_DIR
