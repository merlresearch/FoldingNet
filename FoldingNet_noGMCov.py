# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from FoldingNet_noGM import FoldingNet_noGM


class FoldingNet_noGMCov(FoldingNet_noGM):
    def encoder(self):
        """X -> Code"""
        cc = self.cc
        N = self.data_shape()[-2]
        cc.comment_bar("Encoder")
        cc.silence("Cov")
        self.XKrelu("X", "X1", 64).XKrelu("X1", "X2", 64).XKrelu("X2", "X3", 64).XKrelu("X3", "X4", 128).XK(
            "X4", "X5", 1024
        ).cc.space()  # BxNx1024
        cc.reshape("X5", "X5_B1NK", shape=[0, 1, -1, 1024])  # Bx1xNx1024
        cc.pool("X5_B1NK", "X6", kernel_h=N, kernel_w=1)  # Bx1x1x1024
        self.XK("X6", "X7", 512, axis=1).cc.relu("X7").space()  # Bx512
        self.XK("X7", self.code_name(), self.code_length())  # Bx512


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    FoldingNet_noGMCov(NETWORK_NAME).main()
