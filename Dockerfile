FROM frankierr/openpose_containers:bionic_multi

RUN export LC_ALL=C.UTF-8 && \
    apt-get update && \
    apt-get remove -y cython3 && \
    apt-get install -y --no-install-recommends python3.7-venv

RUN git clone https://github.com/frankier/skelshop/ /opt/skelshop && \
    curl -sSL \
      https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
      | python3.7 || true

RUN python3.7 -m pip install --upgrade pip setuptools

RUN $HOME/.poetry/bin/poetry config virtualenvs.create false

WORKDIR /opt/skelshop

COPY pyproject.toml poetry.lock ./

RUN python3.7 $HOME/.poetry/bin/poetry install && \
    rm -rf /root/.cache

COPY . /opt/skelshop

RUN ./install_rest.sh && \
    snakemake --cores 4
