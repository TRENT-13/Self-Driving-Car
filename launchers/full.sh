#!/bin/bash
set -e
source /environment.sh
dt-launchfile-init

echo "Starting full launcher..."

# Resolve real path of the script, in case it's a symlink
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Now use full real paths to other scripts
dt-exec "$SCRIPT_DIR/display.sh"
dt-exec "$SCRIPT_DIR/utils.sh"

dt-launchfile-join

