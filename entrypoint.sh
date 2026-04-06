#!/bin/sh
set -e

# Always update config from image (deployment manages the source of truth)
mkdir -p /data
cp /app/config.default.json /data/config.json

exec "$@"
