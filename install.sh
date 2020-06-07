#!/usr/bin/env bash

shopt -s extglob
git submodule update --init --recursive
poetry install
export SITE=$(echo $(poetry env list --full-path)/lib/python*/site-packages)
echo "$(pwd)/submodules/" > "$SITE/lighttrack.pth"
echo "$(pwd)/submodules/lighttrack" > "$SITE/lighttrackinner.pth"
echo "$(pwd)/submodules/lighttrack/graph" > "$SITE/lighttrackgraph.pth"
echo "$(pwd)/submodules/opencv_wrapper" > "$SITE/opencv_wrapper.pth"
cd $(pwd)/submodules/ufunclab && poetry run python setup.py install
