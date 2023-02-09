# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import FoldingNetDecoder3D
from FoldingNet import FoldingNet


class FoldingNet3D(FoldingNetDecoder3D, FoldingNet):
    pass


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    FoldingNet3D(NETWORK_NAME).main()
