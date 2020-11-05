FROM frankierr/openpose_containers:bionic_multi

RUN ln -sf /usr/bin/python3.7 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.7 /usr/bin/python

RUN export LC_ALL=C.UTF-8 && \
    apt-get update && \
    apt-get remove -y cython3 && \
    apt-get install -y --no-install-recommends python3.7-venv

RUN python3 -m pip install --upgrade \
    pip==20.2.4 \
    setuptools==50.3.2 \
    poetry==1.1.4

RUN poetry config virtualenvs.create false

WORKDIR /opt/skelshop

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev -E pipeline -E play -E ssmat -E face && \
    rm -rf /root/.cache

COPY . /opt/skelshop

# Install again to get the skelshop package itself
RUN poetry install --no-dev -E pipeline -E play -E ssmat -E face && \
    rm -rf /root/.cache

RUN ./install_rest.sh && \
    python3 -m snakemake --cores 4 && \
    pip cache purge
