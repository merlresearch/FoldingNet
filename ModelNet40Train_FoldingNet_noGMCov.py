# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import ModelNet40Train
from FoldingNet_noGMCov import FoldingNet_noGMCov


class ModelNet40Train_FoldingNet_noGMCov(ModelNet40Train, FoldingNet_noGMCov):
    def n_epoch(self):
        return 400


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    ModelNet40Train_FoldingNet_noGMCov(NETWORK_NAME).main()
