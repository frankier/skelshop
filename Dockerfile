ARG VAR

FROM frankierr/openpose_containers:focal_cpu AS focal_cpu_base

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN apt-get install -y --no-install-recommends unzip libjpeg62-dev

FROM frankierr/openpose_containers:focal_nvcaffe AS focal_nvcaffe_base

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN apt-get install -y --no-install-recommends unzip libjpeg62-dev

RUN cd /opt && \
    git clone --recursive --branch master \
      https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. \
      -DUSE_CUDA=/usr/local/cuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python setup.py install

FROM focal_${VAR}_base

RUN python3 -m pip install --upgrade \
    pip==20.2.4 \
    setuptools==50.3.2 \
    poetry==1.1.4

RUN poetry config virtualenvs.create false

WORKDIR /opt/skelshop

COPY pyproject.toml poetry.lock ./

RUN poetry export \
      --without-hashes \
      -E pipeline \
      -E play \
      -E ssmat \
      -E face \
      -E calibrate \
      -E buildrefs \
      -E clus | \
    sed '/decord/d' > requirements.txt && \
    python3 -m pip install -r requirements.txt && \
    rm requirements.txt && \
    rm -rf /root/.cache

COPY . /opt/skelshop

RUN echo "/opt/skelshop" > \
    /usr/local/lib/python3.8/dist-packages/skelshop.pth

# And reinstall again...(!)
RUN pip install virtualenv && \
    ./install_rest.sh && \
    python3 -m snakemake --cores 4 && \
    rm -rf ~/.cache/pip/

COPY docker/skelshop_env /.skelshop_env
COPY docker/skelshop_entrypoint /.skelshop_entrypoint

ENTRYPOINT ["/usr/bin/bash", "/.skelshop_entrypoint"]
