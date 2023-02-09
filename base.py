# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys

# setup path
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(SCRIPT_FOLDER)
PYCAFFE_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffe/install/python"))
CAFFE_BIN = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffe/install/bin/caffe"))
CAFFECUP_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER, "../caffecup"))
assert os.path.exists(os.path.join(PYCAFFE_FOLDER, "caffe/_caffe.so"))
assert os.path.exists(os.path.join(CAFFECUP_FOLDER, "caffecup/version.py"))
sys.path.insert(0, PYCAFFE_FOLDER)
sys.path.insert(1, CAFFECUP_FOLDER)
sys.path.insert(2, ROOT_FOLDER)
ADDITIONAL_PYTHONPATH_LIST = [ROOT_FOLDER, PYCAFFE_FOLDER]

CWD = os.getcwd()
DATA_FOLDER = os.path.join(CWD, "data")
if not os.path.exists(DATA_FOLDER):
    raise RuntimeError("Please make sure your data folder exists: " + DATA_FOLDER)


class BaseExperiment(object):
    def __init__(self, network_name, network_path=None):
        self.network_name = network_name
        if network_path is None:
            network_path = "./experiments/{}/{}.ptt".format(network_name, network_name)
        self.network_path = network_path
        self.solver_path = os.path.join(os.path.dirname(self.network_path), "solve.ptt")
        self.cc = None  # should be setup by child class

    def data(self):
        raise NotImplementedError()

    def network(self, plot=True):
        raise NotImplementedError()

    def solver(self):
        raise NotImplementedError()

    def design(self):
        self.data()
        self.network()
        self.solver()
        import inspect

        print(
            "Next:\n"
            " 1. check generated files\n"
            " 2. start training (append --dryrun to check; append -h for more help):\n"
            "python %s brew" % (inspect.getmodule(inspect.stack()[2][0]).__file__)
        )

    def brew(self):
        import caffecup.brew.run_solver as run_solver

        args = run_solver.get_args(sys.argv)
        args.caffe = CAFFE_BIN
        args.additional_pythonpath += ADDITIONAL_PYTHONPATH_LIST
        try:
            run_solver.main(args)
        except IOError as e:
            import inspect

            print(e)
            print(
                "Did you forget to execute the following command before brew?\n\tpython %s"
                % (inspect.getmodule(inspect.stack()[2][0]).__file__)
            )

    def test(self):
        import run_test

        run_test.main(run_test.get_args(sys.argv))

    def interp(self):
        import run_interpolate

        run_interpolate.main(run_interpolate.get_args(sys.argv))

    @staticmethod
    def plot():
        import caffecup.viz.plot_traintest_log as plotter

        try:
            args = plotter.get_args(sys.argv)
        except ValueError:
            sys.argv.append(os.path.join(CWD, "logs"))
            args = plotter.get_args(sys.argv)

        args.all_in_one = 1
        if args.axis_left is None:
            args.axis_left = [0.02, 0.05]
        plotter.main(args)

    def main(self):
        if len(sys.argv[1:]) == 0:
            self.design()
        else:
            if sys.argv[1] == "brew":
                sys.argv.pop(1)
                sys.argv.insert(1, "-s")
                sys.argv.insert(2, self.solver_path)
                self.brew()
            elif sys.argv[1] == "test":
                sys.argv.pop(1)
                sys.argv.insert(1, "-s")
                sys.argv.insert(2, self.solver_path)
                sys.argv.insert(3, "--core")
                sys.argv.insert(4, "test")
                sys.argv.append("--wrapper")
                if self.solver_path.find("_noGM") >= 0:
                    if sys.argv.count("--test_dataset") == 0:
                        sys.argv.append("--test_dataset")
                        sys.argv.append("ModelNet40Test")
                self.test()
            elif sys.argv[1] == "svm":
                sys.argv.pop(1)
                sys.argv.insert(1, "-s")
                sys.argv.insert(2, self.solver_path)
                sys.argv.insert(3, "--core")
                sys.argv.insert(4, "svm")
                sys.argv.append("--wrapper")
                if self.solver_path.find("_noGM") >= 0:
                    if sys.argv.count("--test_dataset") == 0:
                        sys.argv.append("--test_dataset")
                        sys.argv.append("ModelNet40Test")
                self.test()
            elif sys.argv[1] == "interp":
                sys.argv.pop(1)
                sys.argv.insert(1, "-s")
                sys.argv.insert(2, self.solver_path)
                sys.argv.insert(3, "--core")
                sys.argv.insert(4, "interp")
                sys.argv.append("--wrapper")
                self.interp()
            elif sys.argv[1] == "clean":
                sys.argv.pop(1)
                sys.argv.insert(1, "-s")
                sys.argv.insert(2, self.solver_path)
                sys.argv.insert(3, "--core")
                sys.argv.insert(4, "test")
                sys.argv.append("--wrapper")
                sys.argv.append("--clean")
                if self.solver_path.find("_noGM") >= 0:
                    if sys.argv.count("--test_dataset") == 0:
                        sys.argv.append("--test_dataset")
                        sys.argv.append("ModelNet40Test")
                self.test()
            else:
                raise NotImplementedError()


if __name__ == "__main__":
    BaseExperiment.plot()
