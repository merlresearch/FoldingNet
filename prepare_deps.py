# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys

import glog as logger

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
CAFFE_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffe"))
CAFFECUP_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffecup"))
CWD = os.getcwd()
assert os.path.samefile(CWD, ROOT_FOLDER)


def prepare_caffe(args):
    os.chdir(ROOT_FOLDER)

    if os.path.exists(CAFFE_FOLDER):
        PYCAFFE_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffe/install/python"))
        CAFFE_BIN = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffe/install/bin/caffe"))
        assert os.path.exists(CAFFE_BIN)
        assert os.path.exists(os.path.join(PYCAFFE_FOLDER, "caffe/_caffe.so"))
        logger.info("caffe installed!")
        return

    # download source
    if not os.path.exists(CAFFE_FOLDER):
        ret = os.system("git clone https://github.com/simbaforrest/caffe.git ../caffe")
        assert ret == 0
        os.chdir(CAFFE_FOLDER)
        ret = os.system("git checkout tags/kcnet -b kcnet")
        assert ret == 0

    # cmake
    os.chdir(CAFFE_FOLDER)
    os.makedirs("build")
    os.chdir("build")
    CUDNN_FOLDER = raw_input("CUDNN Root Folder:")
    ccmake = raw_input("Use ccmake for more configurations? (y/[n])")
    ccmake = True if ccmake == "y" else False
    ret = os.system(
        '{} .. -DALLOW_LMDB_NOLOCK=ON -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="50" -DCUDNN_ROOT="{}" -DCUDNN_INCLUDE="{}" -DCUDNN_LIBRARY="{}"'.format(
            "ccmake" if ccmake else "cmake",
            CUDNN_FOLDER,
            os.path.join(CUDNN_FOLDER, "include"),
            os.path.join(CUDNN_FOLDER, "lib64/libcudnn.so"),
        )
    )
    assert ret == 0

    # build
    ret = os.system("make -j32")
    assert ret == 0

    # install
    ret = os.system("make install -j32")
    assert ret == 0


def prepare_caffecup(args):
    os.chdir(ROOT_FOLDER)

    if os.path.exists(CAFFECUP_FOLDER):
        assert os.path.exists(os.path.join(CAFFECUP_FOLDER, "caffecup/version.py"))
        logger.info("caffecup installed!")
        return
    ret = os.system("git clone https://github.com/simbaforrest/caffecup.git ../caffecup")
    assert ret == 0
    os.chdir(CAFFECUP_FOLDER)
    ret = os.system("git checkout tags/foldingnet -b foldingnet")
    assert ret == 0


def main(args):
    prepare_caffe(args)
    prepare_caffecup(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
