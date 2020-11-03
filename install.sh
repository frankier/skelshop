#!/usr/bin/env bash

poetry install $@
git submodule update --init --recursive
./install_rest.sh
