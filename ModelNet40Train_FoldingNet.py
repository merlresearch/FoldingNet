# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import ModelNet40TrainGraph
from FoldingNet import FoldingNet


class ModelNet40Train_FoldingNet(ModelNet40TrainGraph, FoldingNet):
    def n_epoch(self):
        return 400


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    ModelNet40Train_FoldingNet(NETWORK_NAME).main()
