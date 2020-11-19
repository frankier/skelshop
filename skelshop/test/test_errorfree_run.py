import tempfile
from os.path import join
import os

import requests
from pytest import fixture

import skelshop.config as config
from skelshop.test.testing_conf import TestingConf as testconf
from skelshop.cmd.playsticks import playsticks_fn


@fixture
def conf():
    config.set_conf(testconf)
    return config.conf


@fixture
def test_data(conf):
    res = {}
    tmpdir = tempfile.mkdtemp()
    download_names = {'vid_file': 'shorter_ellen.avi', 'skel_file': 'shorter_ellen_skel.h5'}
    for key, file in download_names.items():
        resp = requests.get(conf.TESTFILES_URL+'/'+file, allow_redirects=True)
        with open(join(tmpdir, file), 'wb') as ofile:
            ofile.write(resp.content)
        res[key] = join(tmpdir, file)
    return res


def test_errorfree_run_drawsticks_cmd(conf, test_data):
    '''cd /home/chris/Documents/JOBS/Uhrig-Gesture-Recog/skelshop && SDL_VIDEODRIVER=dummy;PYTHONUNBUFFERED=1; \
    OPENPOSE=/home/chris/Documents/JOBS/Uhrig-Gesture-Recog/skelshop_deps/openpose_cpu/; \
    LD_LIBRARY_PATH=/home/chris/Documents/JOBS/Uhrig-Gesture-Recog/skelshop_deps/openpose_cpu/build/src/openpose/; \
    PYTHONPATH=/home/chris/Documents/JOBS/Uhrig-Gesture-Recog/skelshop_deps/openpose_cpu/build/python/; \
    MODEL_FOLDER=/home/chris/Documents/JOBS/Uhrig-Gesture-Recog/skelshop_deps/openpose_cpu/models  \
    poetry run python skelshop --ffprobe-bin /usr/bin/ffprobe playsticks \
    /home/chris/Documents/JOBS/Uhrig-Gesture-Recog/data/short_ellen.avi --skel \
    /home/chris/Documents/JOBS/Uhrig-Gesture-Recog/data/short_ellen_bkp.h5'''
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    playsticks_fn(test_data['vid_file'], (test_data['skel_file'], ))



def main():
    conf_ = conf.__wrapped__()
    test_data_ = test_data.__wrapped__(conf_)
    test_errorfree_run_drawsticks_cmd(conf_, test_data_)



if __name__ == '__main__':
    main()
