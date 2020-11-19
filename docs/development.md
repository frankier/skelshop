Please install the [pre-commit](https://pre-commit.com/) based git hooks to run
black and some basic code checks before submitting a PR. For example:

    $ pip install --user pre-commit && pre-commit install

You can also run them manually at any time:

    $ ./run-checks.sh

In case you make changes, but do not have OpenPose installed, don't forget that
you can easily test your changes using Docker, e.g.

    $ DOCKER_TAG=focal_cpu DOCKERFILE_PATH=Dockerfile IMAGE_NAME=focal_cpu
      ./hooks/build

    $ docker run --mount type=bind,source=/,target=/host/  focal_cpu:latest
      python -m skelshop calibrate process-dlib-dir /host/to/dlib/examples/faces/
      /host/to/skelshop/calib.dlib.pqt
