Bootstrap: shub
From: frankier/gsoc2020:frankier_gsoc2020

%post
    export LC_ALL=C.UTF-8
    apt-get install -y --no-install-recommends python3-venv
    git clone https://github.com/frankier/skelshop/ /opt/skelshop
    curl -sSL \
        https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
        | python
    export PATH=$HOME/.poetry/bin/:$PATH
    poetry config virtualenvs.create false
    cd /opt/skelshop && \
        ./install_all.sh && \
        snakemake --cores 4

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export LD_LIBRARY_PATH=$OPENPOSE/src/openpose/:$LD_LIBRARY_PATH
    export MODEL_FOLDER=$OPENPOSE_SRC/models
