# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import ShapeNetPartTrain
from FoldingNet_noGMCov import FoldingNet_noGMCov


class ShapeNetPartTrain_FoldingNet_noGMCov(ShapeNetPartTrain, FoldingNet_noGMCov):
    def n_epoch(self):
        return 330


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    ShapeNetPartTrain_FoldingNet_noGMCov(NETWORK_NAME).main()
