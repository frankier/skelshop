FROM frankierr/openpose_containers:bionic_multi

RUN ln -sf /usr/bin/python3.7 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.7 /usr/bin/python

RUN export LC_ALL=C.UTF-8 && \
    apt-get update && \
    apt-get remove -y cython3 && \
    apt-get install -y --no-install-recommends python3.7-venv

RUN git clone https://github.com/frankier/skelshop/ /opt/skelshop && \
    curl -sSL \
      https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
      | python3 || true

RUN python3 -m pip install --upgrade pip setuptools

RUN $HOME/.poetry/bin/poetry config virtualenvs.create false

WORKDIR /opt/skelshop

COPY pyproject.toml poetry.lock ./

RUN $HOME/.poetry/bin/poetry install && \
    rm -rf /root/.cache

COPY . /opt/skelshop

RUN export PATH=$HOME/.poetry/bin/:$PATH && \
    ./install_rest.sh && \
    python3 -m snakemake --cores 4
