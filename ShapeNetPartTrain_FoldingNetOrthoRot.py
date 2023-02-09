# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import ShapeNetPartTrainOrthoRot
from ShapeNetPartTrain_FoldingNet import ShapeNetPartTrain_FoldingNet


class ShapeNetPartTrain_FoldingNetOrthoRot(ShapeNetPartTrainOrthoRot, ShapeNetPartTrain_FoldingNet):
    def n_epoch(self):
        return 330


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    ShapeNetPartTrain_FoldingNetOrthoRot(NETWORK_NAME).main()
