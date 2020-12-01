# Installing on your local machine

I happened to have a clean install of Ubuntu 20.04, so the following instructions should work. This is for a simple machine wihout an NVIDIA-GPU.

* Frankie's recipe for openpose-CPU: https://github.com/frankier/openpose_containers/blob/main/snippets/get_openpose
* less complete instructions: https://github.com/frankier/openpose_containers/blob/main/snippets/build_op_cpu
* Rest is copied from the dockerfiles

```
export SKELSHOP_DEPS="~/skelshop_deps"
export SKELSHOP_DIR="~/skelshop"
```
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
cd $SKELSHOP_DEPS && \
    git clone \
      https://github.com/CMU-Perceptual-Computing-Lab/openpose.git \
      --branch master --single-branch && \
    cd openpose && \
    git submodule update --recursive --remote && \
    git describe --always > .git-describe
mv $SKELSHOP_DEPS/openpose $SKELSHOP_DEPS/openpose_cpu
cd $SKELSHOP_DEPS/openpose_cpu
wget https://raw.githubusercontent.com/frankier/openpose_containers/main/bionic/respect_mkldnnroot.patch
wget https://raw.githubusercontent.com/frankier/openpose_containers/main/bionic/CMakeLists.patch
git apply CMakeLists.patch respect_mkldnnroot.patch

sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libopencv-dev #500MB!
sudo apt-get install libboost-all-dev #400MB!
apt-get install -y --no-install-recommends build-essential git wget nano dialog software-properties-common libatlas-base-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev
sudo apt-get install --no-install-recommends libgflags-dev libgoogle-glog-dev liblmdb-dev pciutils 
sudo apt-get install --no-install-recommends ocl-icd-opencl-dev libviennacl-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev gfortran pkg-config libssl-dev libcanberra-gtk-module

wget -nv -O - \
        https://apt.kitware.com/keys/kitware-archive-latest.asc | \
        gpg --dearmor - | \
        tee /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get -y --no-install-recommends install cmake

git clone https://github.com/oneapi-src/oneDNN.git && \
    cd oneDNN && \
    git reset --hard ae00102be506ed0fe2099c6557df2aa88ad57ec1 && \
    cd scripts && \
    ./prepare_mkl.sh && \
    cd .. && \
    mkdir -p build && \
    cd build && \
    cmake \
        -DCMAKE_CXX_FLAGS="-w" \
        .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf oneDNN && \
    mkdir -p $SKELSHOP_DEPS/openpose_cpu/build && \
    cd $SKELSHOP_DEPS/openpose_cpu/ && \
    export MKLDNNROOT=/usr/local && \
    cd build && \
    cmake \
        -DGPU_MODE=CPU_ONLY \
        -DBUILD_PYTHON=ON \
        .. && \
    make -j`nproc` && \
    cd .. && \
    rm -rf 3rdparty .git
#worked on 5.11.2020

sudo apt-get install python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev
cd $SKELSHOP_DIR
./install_all.sh 
poetry run snakemake --cores 6

```

## Setting Environment-Variables

```
echo "export SKELSHOP_DIR="path/to/your/skelshop"
export SKELSHOP_DEPS="path/to/your/skelshop_deps"
export OPENPOSE=$SKELSHOP_DEPS/openpose_cpu/; \
export LD_LIBRARY_PATH=$SKELSHOP_DEPS/openpose_cpu/build/src/openpose/; \
export PYTHONPATH=$SKELSHOP_DEPS/openpose_cpu/build/python/; \
export MODEL_FOLDER=$SKELSHOP_DEPS/openpose_cpu/models " >> ~/.bashrc
```

## Running code

Then you should be able to run the code! 

```
cd $SKELSHOP_DIR && \
poetry run python skelshop --ffprobe-bin /usr/bin/ffprobe playsticks \
path/to/your/data/short_ellen.avi --skel \
path/to/your/data/data/short_ellen.h5
```
or
```
PYTHONPATH=$SKELSHOP_DEPS/openpose_cpu/build/python/:$SKELSHOP_DEPS/openpose_cpu/build/python/:$SKELSHOP_DIR:$SKELSHOP_DIR/submodules/lighttrack/graph:$PYTHONPATH poetry run snakemake tracked_all \
  --cores 8 \
  --config \
  VIDEO_BASE=path/to/your/dataset/ \
  DUMP_BASE=path/to/your/dump \
  WORK=path/to/your/work
```
