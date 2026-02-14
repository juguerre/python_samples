#!/bin/bash
# This proxy script is needed in order to setup path environment var prior to `uv run`  execution.
export PATH="$PATH:$HOME/.local/bin"
# echo $PATH
exec "$@" || true

exit 0
