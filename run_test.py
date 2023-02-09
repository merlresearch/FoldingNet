# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys
import time

import caffe
import glog as logger
import numpy as np
from caffecup.brew.run_solver import fastprint, solver2dict
from sklearn.svm import LinearSVC

import base  # need this to setup the right caffe
from utils import dynamic_test_net, dynamic_transfer_net


# http://stackoverflow.com/a/4836734/2303236
def natural_sort(l):
    import re

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


class Tester(object):
    def __init__(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)

    def evaluate_core(self, network_path, weight_path, test_iters, return_code=False, return_label=False):
        net = caffe.Net(network_path, weights=weight_path, phase=caffe.TEST)

        loss = 0
        codewords = []
        labels = []
        for ti in range(test_iters):
            net.forward()
            loss += net.blobs["loss"].data
            if return_code:
                codewords.append(np.array(net.blobs["Code"].data))
            if return_label:
                labels.append(np.array(net.blobs["label"].data))

        ret = {"loss": loss / float(test_iters)}
        if return_code:
            if len(codewords[0].shape) > 1:
                codewords = np.concatenate(codewords)
            else:
                codewords = np.stack(codewords)
            ret.update({"code": codewords})
        if return_label:
            if len(labels[0].shape) > 0:
                labels = np.concatenate(labels)
            else:
                labels = np.stack(labels)
            ret.update({"label": labels})

        return ret

    def evaluate(self, solver_path, test_dataset, clean=False):
        sdict = solver2dict(solver_path)
        snapshot_prefix = sdict["snapshot_prefix"]
        snapshot_folder = os.path.dirname(snapshot_prefix)
        all_files = natural_sort(os.listdir(snapshot_folder))
        network_path = sdict["net"]
        base_classname = os.path.splitext(os.path.basename(network_path))[0]
        test_net = dynamic_test_net(dataset_classname=test_dataset, base_classname=base_classname)
        test_iter = test_net.data_shape()[0]
        min_loss_at, min_loss = -1, np.inf
        all_dict = []
        for fi in all_files:
            if fi.endswith(".caffemodel"):
                fi = os.path.join(snapshot_folder, fi)
                ith_iter = int(fi.replace(snapshot_prefix, "").replace("_iter_", "").replace(".caffemodel", ""))
                ret = self.evaluate_core(test_net.network_path, fi, test_iter)
                all_dict.append((ith_iter, ret))
                fastprint("%6d: %s" % (ith_iter, " ".join(["%s=%.4f" % (k, ret[k]) for k in sorted(ret.keys())])))
                if min_loss > ret["loss"]:
                    min_loss_at = ith_iter
                    min_loss = ret["loss"]
        fastprint("MIN loss (@%6d)=%.4f" % (min_loss_at, min_loss))

        if clean:
            for fi in all_files:
                if fi.endswith(".caffemodel"):
                    fi = os.path.join(snapshot_folder, fi)
                    ith_iter = int(fi.replace(snapshot_prefix, "").replace("_iter_", "").replace(".caffemodel", ""))
                    if ith_iter != min_loss_at:
                        if os.path.exists(fi):
                            os.remove(fi)
                    fi2 = fi.replace(".caffemodel", ".solverstate")
                    if os.path.exists(fi2):
                        os.remove(fi2)

        return all_dict

    def transfer_svm(self, solver_path, train_dataset, test_dataset, weight_path=None):
        sdict = solver2dict(solver_path)
        snapshot_prefix = sdict["snapshot_prefix"]
        if weight_path is None:
            snapshot_folder = os.path.dirname(snapshot_prefix)
            all_files = os.listdir(snapshot_folder)
            assert len(all_files) == 1
            weight_path = os.path.join(snapshot_folder, all_files[0])

        network_path = sdict["net"]
        base_classname = os.path.splitext(os.path.basename(network_path))[0]
        train_net = dynamic_transfer_net(dataset_classname=train_dataset, base_classname=base_classname, no_batch=False)
        assert train_net.data_shape()[0] % train_net.batch_size() == 0
        train_iter = train_net.data_shape()[0] / train_net.batch_size()
        test_net = dynamic_transfer_net(dataset_classname=test_dataset, base_classname=base_classname, no_batch=True)
        test_iter = test_net.data_shape()[0]
        fastprint("train_iter={}".format(train_iter))
        fastprint("test_iter={}".format(test_iter))

        train_ret = self.evaluate_core(
            train_net.network_path, weight_path, train_iter, return_code=True, return_label=True
        )
        test_ret = self.evaluate_core(
            test_net.network_path, weight_path, test_iter, return_code=True, return_label=True
        )

        clf = LinearSVC(random_state=0)
        clf.fit(train_ret["code"], train_ret["label"])
        test_pred = clf.predict(test_ret["code"])
        test_gt = test_ret["label"].flatten()
        accuracy = np.sum(test_pred == test_gt).astype(float) / test_net.data_shape()[0]
        fastprint("transfer linear SVM accuracy={:.4f}".format(accuracy))
        return accuracy


def make_cmd(args):
    if args.jobname == "":
        args.jobname = args.core + "_" + os.path.basename(args.sdict["snapshot_prefix"]) + "_" + args.test_dataset
    if args.srun:
        srun = "srun{} -X -D $PWD --gres gpu:1 ".format(" -p " + args.cluster if args.cluster else "")
        if args.jobname != "none":
            srun += "--job-name=" + args.jobname + " "
    else:
        srun = ""
    cmd = srun

    cmd += 'python {} --core {} -s {} --test_dataset "{}"'.format(
        os.path.basename(__file__), args.core, args.solver, args.test_dataset
    )
    if args.clean:
        cmd += " --clean"

    if args.logname == "":
        args.logname = args.jobname
    logpath = os.path.join(os.path.dirname(args.solver), args.logname + ".txt")
    if os.path.exists(logpath):
        raise RuntimeError("log already existed:" + logpath)
    if args.logname != "none":
        log = " 2>/dev/null | tee {}".format(logpath)
        # log = ' | tee {}'.format(logpath)
    else:
        log = ""
    cmd += log
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
        if args.core == "test" or args.core == "svm":
            args.sdict = solver2dict(args.solver)
            run(args)
        else:
            raise ValueError('unknown args.core="{}"'.format(args.core))
    else:
        if args.core == "test":
            Tester().evaluate(args.solver, test_dataset=args.test_dataset, clean=args.clean)
        elif args.core == "svm":
            test_dataset = args.test_dataset + "WithLabel"
            train_dataset = test_dataset.replace("Test", "Train", 1)
            Tester().transfer_svm(args.solver, train_dataset=train_dataset, test_dataset=test_dataset)
        else:
            raise ValueError('unknown args.core="{}"'.format(args.core))


def get_args(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument("-s", "--solver", type=str, help="path to solver")

    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        default=False,
        help="clean snapshot folder and only keep best weights",
    )

    parser.add_argument("--cluster", type=str, default="", help="which cluster")
    parser.add_argument("--jobname", type=str, default="", help="cluster job name")
    parser.add_argument("--logname", type=str, default="", help="log file name")

    parser.add_argument("--no-srun", dest="srun", action="store_false", default=True, help="DO NOT use srun")

    parser.add_argument("--dryrun", dest="dryrun", action="store_true", default=False, help="sleep 20 seconds")

    parser.add_argument("--wrapper", dest="wrapper", action="store_true", default=False, help="run with wrapper")

    parser.add_argument("--core", type=str, default="test", help="run which core command (test/svm)")

    parser.add_argument(
        "--test_dataset",
        type=str,
        default="ModelNet40TestGraph",
        help="which test data set ([ModelNet40TestGraph]/ModelNet40Test/ShapeNetPartTestGraph/ShapeNetPartTest)",
    )

    args = parser.parse_args(argv[1:])
    args.cwd = os.getcwd()
    args.raw_argv = " ".join(argv)
    args.additional_pythonpath = ["./"]
    return args


if __name__ == "__main__":
    args = get_args(sys.argv)
    main(args)
