ARG BASE
ARG VAR

FROM frankierr/openpose_containers:bionic_multi AS bionic_base

RUN ln -sf /usr/bin/python3.7 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.7 /usr/bin/python

RUN export LC_ALL=C.UTF-8 && \
    apt-get update && \
    apt-get remove -y cython3 && \
    apt-get install -y --no-install-recommends python3.7-venv

FROM frankierr/openpose_containers:focal_${VAR} AS focal_base

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN apt-get install -y --no-install-recommends unzip

FROM ${BASE}_base

RUN apt-get install -y --no-install-recommends ffprobe

RUN python3 -m pip install --upgrade \
    pip==20.2.4 \
    setuptools==50.3.2 \
    poetry==1.1.4

RUN poetry config virtualenvs.create false

WORKDIR /opt/skelshop

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev -E pipeline -E play -E ssmat -E face -E calibrate && \
    rm -rf /root/.cache

COPY . /opt/skelshop

# Install virtualenv because it gets removed for some reason in the previous step
# Install again to get the skelshop package itself
RUN pip install virtualenv && \
    poetry install --no-dev -E pipeline -E play -E ssmat -E face -E calibrate && \
    rm -rf /root/.cache

# And reinstall again...(!)
RUN pip install virtualenv && \
    ./install_rest.sh && \
    python3 -m snakemake --cores 4 && \
    rm -rf ~/.cache/pip/

COPY docker/skelshop_env /.skelshop_env
COPY docker/skelshop_entrypoint /.skelshop_entrypoint

ENTRYPOINT ["/usr/bin/bash", "/.skelshop_entrypoint"]
