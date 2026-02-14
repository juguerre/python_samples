#!/bin/bash
export PATH="$PATH:$HOME/.local/bin"
# echo $PATH
exec "$@" || true

exit 0
