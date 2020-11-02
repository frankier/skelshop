Bootstrap: docker
From: frankierr/openpose_containers:bionic_nvcaffe

%post
    export LC_ALL=C.UTF-8
    apt-get update
    apt-get remove -y cython3
    apt-get install -y --no-install-recommends python3-venv
    git clone https://github.com/frankier/skelshop/ /opt/skelshop
    curl -sSL \
        https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
        | python || true
    export PATH=$HOME/.poetry/bin/:$PATH
    poetry config virtualenvs.create false
    pip install --upgrade pip setuptools
    cd /opt/skelshop && \
        ./install_all.sh && \
        snakemake --cores 4
    rm -rf /root/.cache

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    export OPENPOSE_SRC=/opt/openpose
    export OPENPOSE_VAR=gpu
    export OPENPOSE=$OPENPOSE_SRC/build
    export OPENPOSE_BIN=$OPENPOSE/examples/openpose/openpose.bin
    export PYTHONPATH="$OPENPOSE/python:$PYTHONPATH"
    export OPENPOSE_MODELS=/opt/openpose_models

    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export LD_LIBRARY_PATH=$OPENPOSE/src/openpose/:$LD_LIBRARY_PATH
    export MODEL_FOLDER=$OPENPOSE_MODELS
