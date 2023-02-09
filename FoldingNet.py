# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from common import Adam, FoldingNetDecoder2D, ShapeNetGraph, chamfer_distance


class FoldingNet(ShapeNetGraph, FoldingNetDecoder2D, Adam):
    def distance_loss_func(self):
        return chamfer_distance

    def encoder(self):
        """X -> Code"""
        cc = self.cc
        N = self.data_shape()[-2]
        cc.silence("n_offset")
        cc.comment_bar("Encoder")
        cc.concat(["X", "Cov"], "XCov", axis=-1)
        self.XKrelu("XCov", "X1", 64).XKrelu("X1", "X2", 64).XKrelu("X2", "X3", 64).LocalMaxPooling("X3", "X3m").XKrelu(
            "X3m", "X4", 128
        ).LocalMaxPooling("X4", "X4m").XK(
            "X4m", "X5", 1024
        ).cc.space()  # (B*N)x1024
        cc.reshape("X5", "X5_B1NK", shape=[self.batch_size(), 1, -1, 1024])  # Bx1xNx1024
        cc.pool("X5_B1NK", "X6", kernel_h=N, kernel_w=1)  # Bx1x1x1024
        self.XK("X6", "X7", 512, axis=1).cc.relu("X7").space()  # Bx512
        self.XK("X7", self.code_name(), self.code_length())  # Bx512


if __name__ == "__main__":
    NETWORK_NAME = os.path.splitext(__file__)[0]
    FoldingNet(NETWORK_NAME).main()
