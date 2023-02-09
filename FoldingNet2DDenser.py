# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from FoldingNet import FoldingNet


class FoldingNet2DDenser(FoldingNet):
    def get_nd_grid_resolution(self):
        return [50, 50]


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    FoldingNet2DDenser(NETWORK_NAME).main()
