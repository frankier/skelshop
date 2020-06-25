#!/usr/bin/env bash

shopt -s extglob
git submodule update --init --recursive
# Tensorflow has over restrictive dependency versions
# Install before and then our own versions get to overwrite its
poetry run pip install tensorflow==2.2.0
poetry install $@

PKGS="$(poetry env list --full-path | cut -d' ' -f1)"
if [ -z "$PKGS" ]; then
    SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
else
    SITE="$(echo $PKGS/lib/python*/site-packages)"
fi
echo "$(pwd)/submodules/" > "$SITE/lighttrack.pth"
echo "$(pwd)/submodules/lighttrack" > "$SITE/lighttrackinner.pth"
echo "$(pwd)/submodules/lighttrack/graph" > "$SITE/lighttrackgraph.pth"
echo "$(pwd)/submodules/opencv_wrapper" > "$SITE/opencv_wrapper.pth"
cd $(pwd)/submodules/ufunclab && poetry run python setup.py install
