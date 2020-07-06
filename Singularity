Bootstrap: shub
From: frankier/gsoc2020:frankier_gsoc2020

%post
    export LC_ALL=C.UTF-8
    apt-get remove -y cython3
    apt-get install -y --no-install-recommends python3-venv
    git clone https://github.com/frankier/skelshop/ /opt/skelshop
    curl -sSL \
        https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
        | python || true
    export PATH=$HOME/.poetry/bin/:$PATH
    poetry config virtualenvs.create false
    cd /opt/skelshop && \
        FORCE_CUDA=1 \
        TORCH_CUDA_ARCH_LIST='3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;3.5+PTX;3.7+PTX;5.0+PTX;5.2+PTX;5.3+PTX;6.0+PTX;6.1+PTX;6.2+PTX;7.0+PTX;7.2+PTX;7.5+PTX' \
        ./install_all.sh && \
        snakemake --cores 4

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export LD_LIBRARY_PATH=$OPENPOSE/src/openpose/:$LD_LIBRARY_PATH
    export MODEL_FOLDER=$OPENPOSE_SRC/models
