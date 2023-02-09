# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import glob
import os
import sys
import time

import caffe
import glog as logger
import numpy as np
from caffecup.brew.run_solver import create_if_not_exist, fastprint, solver2dict

import base  # need this to setup the right caffe
import visualize as vis
from utils import dynamic_decoder_net, dynamic_encoder_net


class Interpolator(object):
    def __init__(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)

    @staticmethod
    def get_output_filename(solver_path, test_dataset, aid, bid, use_dense_decoder):
        interp_folder = os.path.join(
            os.path.dirname(solver_path), "{}interp{}".format("dense_" if use_dense_decoder else "", test_dataset)
        )
        create_if_not_exist(interp_folder, "interpolation")
        interp_output_file = os.path.join(interp_folder, "interp_{}_{}.npz".format(aid, bid))
        return interp_output_file

    def interpolate(self, solver_path, aid, bid, n_interp, test_dataset, weights_path=None, use_dense_decoder=False):
        sdict = solver2dict(solver_path)
        snapshot_folder = os.path.dirname(sdict["snapshot_prefix"])
        interp_output_file = Interpolator.get_output_filename(solver_path, test_dataset, aid, bid, use_dense_decoder)
        if os.path.exists(interp_output_file):
            fastprint("existed:" + interp_output_file)
            return interp_output_file

        if weights_path is None or weights_path == "":
            all_files = os.listdir(snapshot_folder)
            assert len(all_files) == 1
            weights_path = os.path.join(snapshot_folder, all_files[0])
            assert os.path.exists(weights_path)

        network_path = sdict["net"]
        base_classname = os.path.splitext(os.path.basename(network_path))[0]

        encoder = dynamic_encoder_net(dataset_classname=test_dataset, base_classname=base_classname)
        decoder = dynamic_decoder_net(
            base_classname=base_classname, n_XK=encoder.cc.n_XK, use_dense_decoder=use_dense_decoder
        )
        encoder_net = caffe.Net(encoder.network_path, weights=weights_path, phase=caffe.TEST)
        decoder_net = caffe.Net(decoder.network_path, weights=weights_path, phase=caffe.TEST)

        encoder_net.layers[0].old_ith = -1  # prevent reshuffle
        encoder_net.layers[0].ith = aid
        encoder_net.forward()
        code_a = np.array(encoder_net.blobs["Code"].data)
        X_a = np.array(encoder_net.blobs["X"].data[0])

        decoder_net.blobs["Code"].data[...] = code_a
        decoder_net.forward()
        Xp_a = np.array(decoder_net.blobs["Xp"].data[0])

        encoder_net.layers[0].old_ith = -1
        encoder_net.layers[0].ith = bid
        encoder_net.forward()
        code_b = np.array(encoder_net.blobs["Code"].data)
        X_b = np.array(encoder_net.blobs["X"].data[0])

        decoder_net.blobs["Code"].data[...] = code_b
        decoder_net.forward()
        Xp_b = np.array(decoder_net.blobs["Xp"].data[0])

        all_Xp = []
        delta = 1.0 / (n_interp + 1)  # n interpolation points => n+1 sections
        for i in xrange(n_interp):
            ratio = (i + 1) * delta
            code = code_a * (1.0 - ratio) + code_b * ratio
            decoder_net.blobs["Code"].data[...] = code
            decoder_net.forward()
            Xp = np.array(decoder_net.blobs["Xp"].data)
            all_Xp.append(Xp)
        all_Xp = np.concatenate(all_Xp)
        np.savez(interp_output_file, all_Xp=all_Xp, Xa=X_a, Xb=X_b, Xpa=Xp_a, Xpb=Xp_b)
        return interp_output_file


def make_cmd(args):
    if args.jobname == "":
        if args.core == "interp":
            args.jobname = args.core
    if args.srun:
        srun = "srun{} -X -D $PWD --gres gpu:1 ".format(" -p " + args.cluster if args.cluster else "")
        if args.jobname != "none":
            srun += "--job-name=" + args.jobname + " "
    else:
        srun = ""
    cmd = srun

    cmd += "python {} --core {} -s {}".format(os.path.basename(__file__), args.core, args.solver)
    if args.test_dataset:
        cmd += " --test_dataset {}".format(args.test_dataset)
    if args.weight:
        cmd += " --weight {}".format(args.weight)
    if args.dense:
        cmd += " --dense"

    cmd += " --aid {} --bid {} --n_interp {} 2>/dev/null".format(args.aid, args.bid, args.n_interp)
    return cmd


def run(args):
    cmd = make_cmd(args)
    args.cmd = cmd
    logger.info("to run:")
    fastprint(cmd)
    if args.dryrun:
        logger.info("dryrun: sleep 20")
        cmd = "echo PYTHONPATH=$PYTHONPATH; for i in {1..20}; do echo $i; sleep 1; done;"

    my_env = os.environ.copy()
    from base import ADDITIONAL_PYTHONPATH_LIST

    if my_env.has_key("PYTHONPATH"):
        my_env["PYTHONPATH"] = (
            ":".join(args.additional_pythonpath + ADDITIONAL_PYTHONPATH_LIST) + ":" + my_env["PYTHONPATH"]
        )
    else:
        my_env["PYTHONPATH"] = ":".join(args.additional_pythonpath + ADDITIONAL_PYTHONPATH_LIST)
    import subprocess

    THE_JOB = subprocess.Popen(cmd, shell=True, cwd=args.cwd, env=my_env)

    while True:
        retcode = THE_JOB.poll()
        if retcode is not None:
            logger.info("job({}) finished!".format(args.jobname))
            break

        try:
            time.sleep(1)
        except KeyboardInterrupt:
            THE_JOB.kill()
            logger.info("job({}) killed by CTRL-C!".format(args.jobname))
            break


def main(args):
    if args.wrapper:
        if args.core == "interp":
            if args.aid < 0 or args.bid < 0:
                fastprint("please specify a valid id for interpolation!")
                exit(-1)
            run(args)
            if args.vis:
                fname = Interpolator.get_output_filename(args.solver, args.test_dataset, args.aid, args.bid, args.dense)
                if not args.dense:
                    vis.interactive_render_interploation(fname)
        else:
            raise ValueError('unknown args.core="{}"'.format(args.core))
    else:
        if args.core == "interp":
            Interpolator().interpolate(
                solver_path=args.solver,
                aid=args.aid,
                bid=args.bid,
                n_interp=args.n_interp,
                test_dataset=args.test_dataset,
                weights_path=args.weight,
                use_dense_decoder=args.dense,
            )
        else:
            raise ValueError('unknown args.core="{}"'.format(args.core))


def get_args(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument("-s", "--solver", type=str, help="path to solver")

    parser.add_argument("--cluster", type=str, default="", help="which cluster")
    parser.add_argument("--jobname", type=str, default="", help="cluster job name")

    parser.add_argument("--no-srun", dest="srun", action="store_false", default=True, help="DO NOT use srun")

    parser.add_argument("--dryrun", dest="dryrun", action="store_true", default=False, help="sleep 20 seconds")

    parser.add_argument("--wrapper", dest="wrapper", action="store_true", default=False, help="run with wrapper")

    parser.add_argument("--dense", dest="dense", action="store_true", default=False, help="use dense decoder or not")

    parser.add_argument("--vis", dest="vis", action="store_true", default=False, help="use draw interpolation or not")

    parser.add_argument("--core", type=str, default="interp", help="run which core command (interp)")

    parser.add_argument(
        "--test_dataset",
        type=str,
        default="",
        help="which test data set ([]/ModelNet40TestGraph/" + "ModelNet40Test/ShapeNetPartTestGraph/ShapeNetPartTest)",
    )

    parser.add_argument("--n_interp", type=int, default=20, help="number of interpolation points")
    parser.add_argument("--aid", type=int, default=-1, help="id of the first object (interp)")
    parser.add_argument("--bid", type=int, default=-1, help="id of the second object (interp)")
    parser.add_argument("--weight", type=str, default="", help="using which weight to interpolate")

    args = parser.parse_args(argv[1:])
    args.cwd = os.getcwd()
    args.raw_argv = " ".join(argv)
    args.additional_pythonpath = ["./"]
    return args


if __name__ == "__main__":
    args = get_args(sys.argv)
    main(args)
