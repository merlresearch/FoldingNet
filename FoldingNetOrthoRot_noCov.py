# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import ShapeNetGraphOrthoRot
from FoldingNet_noCov import FoldingNet_noCov


class FoldingNetOrthoRot_noCov(ShapeNetGraphOrthoRot, FoldingNet_noCov):
    def n_epoch(self):
        return 330


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    FoldingNetOrthoRot_noCov(NETWORK_NAME).main()
