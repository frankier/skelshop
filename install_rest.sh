#!/usr/bin/env bash

set -o xtrace
set -euo pipefail

shopt -s extglob

PKGS="$(poetry env list --full-path | cut -d' ' -f1)"
if [ -z "$PKGS" ]; then
    SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
else
    SITE="$(echo $PKGS/lib/python*/site-packages)"
fi
echo "$(pwd)/submodules/" > "$SITE/lighttrack.pth"
echo "$(pwd)/submodules/lighttrack" > "$SITE/lighttrackinner.pth"
echo "$(pwd)/submodules/lighttrack/graph" > "$SITE/lighttrackgraph.pth"
echo "$(pwd)/submodules/lighttrack/graph/torchlight" > "$SITE/torchlight.pth"

poetry run pip install -e $(pwd)/submodules/opencv_wrapper
poetry run pip install -e $(pwd)/submodules/ufunclab
FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST='3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;3.5+PTX;3.7+PTX;5.0+PTX;5.2+PTX;5.3+PTX;6.0+PTX;6.1+PTX;6.2+PTX;7.0+PTX;7.2+PTX;7.5+PTX' \
    poetry run pip install -e $(pwd)/submodules/mmskeleton
