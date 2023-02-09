# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from string import Template

import caffecup
import caffecup.brew
import glog as logger
import numpy as np

from base import BaseExperiment

###########################################################################


def pairwise_difference(
    cc,
    input_X,  # BxNxK
    input_Y,  # BxMxK
    output_V,  # BxNxMxK
    name="",
):
    assert isinstance(cc, caffecup.Designer)
    shapeX = np.asarray(cc.shape_of(input_X))
    shapeY = np.asarray(cc.shape_of(input_Y))
    assert np.all(shapeX[:-2] == shapeY[:-2])
    assert shapeX[-1] == shapeY[-1]

    N = shapeX[-2]
    M = shapeY[-2]
    K = shapeX[-1]

    if hasattr(cc, "n_pairwise_difference"):
        cc.n_pairwise_difference += 1
    else:
        cc.n_pairwise_difference = 1

    if name == "":
        name = "pairwise_difference{:d}".format(cc.n_pairwise_difference)

    cc.comment("{}: {} - {}".format(name, input_X, input_Y))

    shapeXr = shapeX[:-2].tolist() + [N, 1, K]
    cc.reshape(input_X, input_X + "_BN1K", shape=shapeXr, name=name + "_X_BN1K")
    cc.tile(input_X + "_BN1K", input_X + "_BNMK", axis=-2, tiles=M, name=name + "_X_BNMK")

    shapeYr = shapeY[:-2].tolist() + [1, M, K]
    cc.reshape(input_Y, input_Y + "_B1MK", shape=shapeYr, name=name + "_Y_B1MK")
    cc.tile(input_Y + "_B1MK", input_Y + "_BNMK", axis=-3, tiles=N, name=name + "_Y_BNMK")

    cc.eltsub(input_X + "_BNMK", input_Y + "_BNMK", output_V, name=name + "_X-Y")
    return cc


def distance_matrix(
    cc,
    input_X,  # BxNxK
    input_Y,  # BxMxK
    output_D,  # BxNxM
    eps=1e-6,
):
    assert isinstance(cc, caffecup.Designer)
    pairwise_difference(cc, input_X, input_Y, output_D + "_difference")  # BxNxMxK
    cc.pow(output_D + "_difference", output_D + "_difference2", power=2)
    # cc.reduce(output_D+'_difference2',output_D+'_squared',axis=-1) #BxNxM
    ##note: the above reduce BNMK to BNM is very slow since the outer loop of reduce is implemented in CPU in caffe
    ##use constant FC layer to achieve the same effect but much faster
    cc.XK(
        output_D + "_difference2",
        output_D + "_squared_BNM1",
        num_output=1,
        axis=-1,
        weight_filler_str=caffecup.learable.filler_constant(1.0),
        weight_param_str=caffecup.learable.learning_param(0, 0),
        bias_term=False,
    )  # BxNxMx1
    cc.reshape(output_D + "_squared_BNM1", output_D + "_squared", shape=[0, 0, 0])  # BxNxM
    cc.pow(output_D + "_squared", output_D, power=0.5, shift=eps)
    return cc


def chamfer_distance(cc, input_D, output_d, output_max_or_mean="max"):  # BxNxM  # B
    assert isinstance(cc, caffecup.Designer)
    shapeD = np.asarray(cc.shape_of(input_D))
    assert len(shapeD) >= 2
    N = shapeD[-2]
    M = shapeD[-1]

    cc.comment("chamfer_distance {}=>{}".format(input_D, output_d))
    cc.pow(input_D, input_D + "_neg", scale=-1)
    cc.reshape(input_D + "_neg", input_D + "_B1NM", shape=[np.prod(shapeD[:-2]), 1] + shapeD[-2:].tolist())  # Bx1xNxM
    cc.pool(
        input_D + "_B1NM", input_D + "_rowMin", kernel_h=N, kernel_w=1, name="RowMAX[%s]" % (input_D + "_B1NM")
    )  # Bx1x1xM
    cc.pool(
        input_D + "_B1NM", input_D + "_colMin", kernel_h=1, kernel_w=M, name="ColMAX[%s]" % (input_D + "_B1NM")
    )  # Bx1xNx1
    cc.global_average_pooling(input_D + "_rowMin", output_d + "_row")  # Bx1x1x1
    cc.global_average_pooling(input_D + "_colMin", output_d + "_col")  # Bx1x1x1
    cc.pow(output_d + "_row", output_d + "_row_pos", scale=-1)
    cc.pow(output_d + "_col", output_d + "_col_pos", scale=-1)

    cc.concat([output_d + "_row_pos", output_d + "_col_pos"], output_d + "_rowcol", axis=-1)  # Bx1x1x2
    if output_max_or_mean.lower() == "max":
        cc.pool(output_d + "_rowcol", output_d + "_B111", kernel_h=1, kernel_w=2)  # Bx1x1x1
    else:
        cc.global_average_pooling(output_d + "_rowcol", output_d + "_B111")  # Bx1x1x1
    if len(shapeD) == 2:
        cc.reshape(
            output_d + "_B111",
            output_d,
            shape=[
                1,
            ],
        )  # B==1
    else:
        cc.reshape(output_d + "_B111", output_d, shape=shapeD[:-2].tolist())  # B
    return cc


def hausdorff_distance(
    cc,
    input_D,  # BxNxM
    output_d,  # B
):
    assert isinstance(cc, caffecup.Designer)
    shapeD = np.asarray(cc.shape_of(input_D))
    assert len(shapeD) >= 2
    N = shapeD[-2]
    M = shapeD[-1]

    cc.comment("hausdorff_distance {}=>{}".format(input_D, output_d))
    cc.pow(input_D, input_D + "_neg", scale=-1)
    cc.reshape(input_D + "_neg", input_D + "_B1NM", shape=[np.prod(shapeD[:-2]), 1] + shapeD[-2:].tolist())  # Bx1xNxM
    cc.pool(
        input_D + "_B1NM", input_D + "_rowMin", kernel_h=N, kernel_w=1, name="RowMAX[%s]" % (input_D + "_B1NM")
    )  # Bx1x1xM
    cc.pool(
        input_D + "_B1NM", input_D + "_colMin", kernel_h=1, kernel_w=M, name="ColMAX[%s]" % (input_D + "_B1NM")
    )  # Bx1xNx1
    cc.pow(input_D + "_rowMin", output_d + "_rowMin_pos", scale=-1)
    cc.pow(input_D + "_colMin", output_d + "_colMin_pos", scale=-1)
    cc.pool(output_d + "_rowMin_pos", output_d + "_row", kernel_h=1, kernel_w=M)  # Bx1x1x1
    cc.pool(output_d + "_colMin_pos", output_d + "_col", kernel_h=N, kernel_w=1)  # Bx1x1x1

    cc.concat([output_d + "_row", output_d + "_col"], output_d + "_rowcol", axis=-1)  # Bx1x1x2
    cc.pool(output_d + "_rowcol", output_d + "_B111", 1, 2)  # Bx1x1x1
    if len(shapeD) == 2:
        cc.reshape(
            output_d + "_B111",
            output_d,
            shape=[
                1,
            ],
        )  # B==1
    else:
        cc.reshape(output_d + "_B111", output_d, shape=shapeD[:-2].tolist())  # B
    return cc


def GlobalMaxPooling(cc, input, output, input_offset="n_offset", name="GlobalMaxPool"):
    assert isinstance(cc, caffecup.Designer)
    shape = cc.shape_of(input)
    shape_offset = cc.shape_of(input_offset)
    shape[0] = shape_offset[0]
    assert cc.register_new_blob(output, shape)

    s = Template(
        """layer {
  name: "$name" type: "GraphPooling"
  graph_pooling_param { mode: MAX }
  bottom: "$input"
  bottom: "$input_offset"
  propagate_down: true
  propagate_down: false
  top: "$output"
}
"""
    )
    cc.fp.write(s.substitute(locals()))
    return cc


def LocalGraphPooling(cc, input, output, mode, name, G_indptr="G_indptr", G_indices="G_indices", G_data="G_data"):
    assert mode in ["MAX", "AVE"]
    assert isinstance(cc, caffecup.Designer)
    cc.shape_of(G_indptr)
    cc.shape_of(G_indices)
    assert cc.register_new_blob(output, cc.shape_of(input))

    s = Template(
        """layer {
  name: "$name" type: "GraphPooling"
  graph_pooling_param { mode: $mode }
  bottom: "$input"
  bottom: "$G_indptr" bottom: "$G_indices" bottom: "$G_data"
  propagate_down: true
  propagate_down: false propagate_down: false propagate_down: false
  top: "$output"
}
"""
    )
    cc.fp.write(s.substitute(locals()))
    return cc


def LocalMaxPooling(cc, input, output, G_indptr="G_indptr", G_indices="G_indices", G_data="G_data", name=""):
    assert isinstance(cc, caffecup.Designer)
    if not hasattr(cc, "n_LMP"):
        cc.n_LMP = 1
    else:
        cc.n_LMP += 1
    if name == "":
        name = "LocalMaxPool{:d}".format(cc.n_LMP)

    return LocalGraphPooling(
        cc, input, output, G_indptr=G_indptr, G_indices=G_indices, G_data=G_data, mode="MAX", name=name
    )


###########################################################################


class BaseData(BaseExperiment):
    def data(self):
        total = self.data_shape()[0] * self.n_epoch()
        if total % self.batch_size() != 0:
            logger.warn(
                "Dataset size ({:d}*{:d}) is not divisible by batch size ({:d})!".format(
                    self.n_epoch(), self.data_shape()[0], self.batch_size()
                )
            )

        self.cc = caffecup.Designer(self.network_path)
        self.cc.name(self.network_name)
        self.cc.comment_bar("Data")
        self.cc.pydata(
            outputs=self.outputs(),
            shapes=self.output_shapes(),
            module="io_layers",
            layer=self.layer_name(),
            param_str=self.param_str(),
            phase="",
        )

    # def n_cls(self): raise NotImplementedError()
    def n_epoch(self):
        return 300

    def outputs(self):
        raise NotImplementedError()

    def data_shape(self):
        raise NotImplementedError()

    def batch_size(self):
        raise NotImplementedError()

    def output_shapes(self):
        raise NotImplementedError()

    def param_str(self):
        raise NotImplementedError()

    def layer_name(self):
        raise NotImplementedError()


class BaseNPYData(BaseData):
    def layer_name(self):
        return "InputXYZCovNPYLayer"

    def outputs(self):
        return ["X", "Cov"]

    def output_shapes(self):
        X_shape = self.data_shape()
        X_shape[0] = self.batch_size()
        Cov_shape = np.array(X_shape)
        Cov_shape[-1] = 9
        return [X_shape, Cov_shape.tolist()]


class BaseGraphData(BaseData):
    def outputs(self):
        return ["X", "Cov", "n_offset", "G_indptr", "G_indices", "G_data"]

    def output_shapes(self):
        X_shape = [self.batch_size() * self.data_shape()[-2], self.data_shape()[-1]]
        Cov_shape = np.array(X_shape)
        Cov_shape[-1] = 9
        return [
            X_shape,
            Cov_shape.tolist(),
            [
                self.batch_size(),
            ],
            [
                X_shape[0] + 1,
            ],
            [
                "nnz",
            ],
            [
                "nnz",
            ],
        ]


class BaseNPYGraphData(BaseGraphData):
    def layer_name(self):
        return "InputXYZCovGraphNPYLayer"


class BaseLMDBGraphData(BaseGraphData):
    def layer_name(self):
        return "InputXYZCovGraphLMDBLayer"


################################### ShapeNet
class ShapeNet(BaseData):
    def data_shape(self):
        return [57448, 2048, 3]

    def batch_size(self):
        return 16  # 57448=8*43*167

    def outputs(self):
        return ["X", "Cov"]

    def layer_name(self):
        return "IOShapeNetCovLMDBLayer"

    def output_shapes(self):
        X_shape = self.data_shape()
        X_shape[0] = self.batch_size()
        Cov_shape = np.array(X_shape)
        Cov_shape[-1] = 9
        return [X_shape, Cov_shape.tolist()]

    def param_str(self):
        return "{'source':'data/shapenet57448xyzonly_16nn_GM_conv.lmdb', 'batch_size':%d}" % (self.batch_size())


class ShapeNetOrthoRot(ShapeNet):
    def param_str(self):
        return (
            "{'source':'data/shapenet57448xyzonly_16nn_GM_conv.lmdb', 'batch_size':%d, 'rand_rotation':'ortho', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class ShapeNetRandRot(ShapeNet):
    def param_str(self):
        return (
            "{'source':'data/shapenet57448xyzonly_16nn_GM_conv.lmdb', 'batch_size':%d, 'rand_rotation':'full', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class ShapeNetGraph(BaseLMDBGraphData, ShapeNet):
    pass


class ShapeNetGraphOrthoRot(BaseLMDBGraphData, ShapeNetOrthoRot):
    pass


class ShapeNetGraphRandRot(BaseLMDBGraphData, ShapeNetRandRot):
    pass


################################### ModelNet40
class ModelNet40Train(BaseNPYData):
    def data_shape(self):
        return [9840, 2048, 3]

    def batch_size(self):
        return 16

    def param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM_cov.npy', 'batch_size':%d}" % (self.batch_size())


class ModelNet40TrainOrthoRot(ModelNet40Train):
    def param_str(self):
        return (
            "{'source':'data/modelNet40_train_file_16nn_GM_cov.npy', 'batch_size':%d, 'rand_rotation':'ortho', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class ModelNet40TrainRandRot(ModelNet40Train):
    def param_str(self):
        return (
            "{'source':'data/modelNet40_train_file_16nn_GM_cov.npy', 'batch_size':%d, 'rand_rotation':'full', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class ModelNet40TrainWithLabel(ModelNet40Train):
    def outputs(self):
        ret = super(ModelNet40Train, self).outputs()
        ret.append("label")
        return ret

    def output_shapes(self):
        ret = super(ModelNet40Train, self).output_shapes()
        ret.append(
            [
                self.batch_size(),
            ]
        )
        return ret

    def param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM_cov.npy', 'batch_size':%d, 'output_label':True}" % (
            self.batch_size()
        )


class ModelNet40Test(BaseNPYData):
    def data_shape(self):
        return [2468, 2048, 3]

    def batch_size(self):
        return 16

    def param_str(self):
        return "{'source':'data/modelNet40_test_file_16nn_GM_cov.npy', 'batch_size':%d}" % (self.batch_size())


class ModelNet40TestWithLabel(ModelNet40Test):
    def outputs(self):
        ret = super(ModelNet40Test, self).outputs()
        ret.append("label")
        return ret

    def output_shapes(self):
        ret = super(ModelNet40Test, self).output_shapes()
        ret.append(
            [
                self.batch_size(),
            ]
        )
        return ret

    def param_str(self):
        return "{'source':'data/modelNet40_test_file_16nn_GM_cov.npy', 'batch_size':%d, 'output_label':True}" % (
            self.batch_size()
        )


class ModelNet40TrainGraph(BaseNPYGraphData, ModelNet40Train):
    pass


class ModelNet40TrainGraphOrthoRot(BaseNPYGraphData, ModelNet40TrainOrthoRot):
    pass


class ModelNet40TrainGraphRandRot(BaseNPYGraphData, ModelNet40TrainRandRot):
    pass


class ModelNet40TrainGraphWithLabel(ModelNet40TrainGraph):
    def outputs(self):
        ret = super(ModelNet40TrainGraph, self).outputs()
        ret.append("label")
        return ret

    def output_shapes(self):
        ret = super(ModelNet40TrainGraph, self).output_shapes()
        ret.append(
            [
                self.batch_size(),
            ]
        )
        return ret

    def param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM_cov.npy', 'batch_size':%d, 'output_label':True}" % (
            self.batch_size()
        )


class ModelNet40TestGraph(BaseNPYGraphData, ModelNet40Test):
    pass


class ModelNet40TestGraphWithLabel(ModelNet40TestGraph):
    def outputs(self):
        ret = super(ModelNet40TestGraph, self).outputs()
        ret.append("label")
        return ret

    def output_shapes(self):
        ret = super(ModelNet40TestGraph, self).output_shapes()
        ret.append(
            [
                self.batch_size(),
            ]
        )
        return ret

    def param_str(self):
        return "{'source':'data/modelNet40_test_file_16nn_GM_cov.npy', 'batch_size':%d, 'output_label':True}" % (
            self.batch_size()
        )


################################### Primitive
class PrimitiveTrain(BaseNPYData):
    def data_shape(self):
        return [4, 2048, 3]

    def batch_size(self):
        return 4

    def param_str(self):
        return "{'source':'data/primitives_tori.npy', 'batch_size':%d}" % (self.batch_size())


class PrimitiveTrainOrthoRot(PrimitiveTrain):
    def param_str(self):
        return (
            "{'source':'data/primitives_tori.npy', 'batch_size':%d, 'rand_rotation':'ortho', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class PrimitiveTrainRandRot(PrimitiveTrain):
    def param_str(self):
        return (
            "{'source':'data/primitives_tori.npy', 'batch_size':%d, 'rand_rotation':'full', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


################################### ShapeNetPart
class ShapeNetPartTrain(BaseNPYData):
    def data_shape(self):
        return [14007, 2048, 3]

    def batch_size(self):
        return 16

    def param_str(self):
        return "{'source':'data/shapenet_part_train_file_16nn_GM_cov.npy', 'batch_size':%d}" % (self.batch_size())


class ShapeNetPartTrainOrthoRot(ShapeNetPartTrain):
    def param_str(self):
        return (
            "{'source':'data/shapenet_part_train_file_16nn_GM_cov.npy', 'batch_size':%d, 'rand_rotation':'ortho', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class ShapeNetPartTrainRandRot(ShapeNetPartTrain):
    def param_str(self):
        return (
            "{'source':'data/shapenet_part_train_file_16nn_GM_cov.npy', 'batch_size':%d, 'rand_rotation':'full', 'rand_online':True, 'random_seed':127482}"
            % (self.batch_size())
        )


class ShapeNetPartTest(BaseNPYData):
    def data_shape(self):
        return [2874, 2048, 3]

    def batch_size(self):
        return 16

    def param_str(self):
        return "{'source':'data/shapenet_part_test_file_16nn_GM_cov.npy', 'batch_size':%d}" % (self.batch_size())


class ShapeNetPartTrainGraph(BaseNPYGraphData, ShapeNetPartTrain):
    pass


class ShapeNetPartTrainGraphOrthoRot(BaseNPYGraphData, ShapeNetPartTrainOrthoRot):
    pass


class ShapeNetPartTrainGraphRandRot(BaseNPYGraphData, ShapeNetPartTrainRandRot):
    pass


class ShapeNetPartTestGraph(BaseNPYGraphData, ShapeNetPartTest):
    pass


###########################################################################


class Adam(BaseData):
    def base_lr(self):
        return 1e-4

    def snapshot_interval(self):
        return int(0.01 * self.data_shape()[0] * self.n_epoch() / (self.batch_size()))

    def solver(self):
        tts = caffecup.brew.TrainSolver(self.solver_path)
        cc = self.cc
        assert isinstance(cc, caffecup.Designer)

        tts.build(
            train_net=cc.filepath,
            n_train_data=self.data_shape()[0],
            train_batch_size=self.batch_size(),
            n_epoch=self.n_epoch(),
            base_lr=self.base_lr(),
            solver_type="Adam",
            lr_policy="fixed",
            snapshot_folder=os.path.join(cc.filedir, "snapshot"),
            snapshot_interval=self.snapshot_interval(),
            weight_decay=1e-6,
        )
        print("written:" + tts.filepath)


###########################################################################


class BaseAutoEncoder(BaseData):
    def code_name(self):
        return "Code"

    def code_length(self):
        return 512

    def reconstructed_name(self):
        return "Xp"

    def XK(self, i, o, n, axis=-1, W_name=None, b_name=None):
        if W_name is None:
            W_name = "W[{}=>{}]".format(i, o)
        if b_name is None:
            b_name = "b[{}=>{}]".format(i, o)
        # layer_name = '{}=>{}'.format(i,o)
        self.cc.XK(
            i,
            o,
            n,
            axis=axis,
            weight_param_str=caffecup.learable.learning_param(name=W_name),
            bias_param_str=caffecup.learable.learning_param(name=b_name, decay_mult=0),
            # name=layer_name
        )
        return self

    def XKrelu(self, i, o, n, **kwargs):
        self.XK(i, o, n, **kwargs).cc.relu(o).space()
        return self

    def LocalMaxPooling(self, i, o):
        assert set(self.outputs()).issuperset(["G_indptr", "G_indices", "G_data"])
        LocalMaxPooling(self.cc, i, o)
        return self

    def network(self, plot=True):
        """X -> loss"""
        self.encoder()
        self.decoder()
        self.loss(plot=plot)

    def encoder(self):
        raise NotImplementedError()

    def decoder(self):
        raise NotImplementedError()

    def distance_loss_func(self):
        raise NotImplementedError()

    def loss(self, plot=True):
        import caffecup.viz.draw_net as ccdraw

        cc = self.cc
        cc.comment_bar("Loss")
        B = self.batch_size()
        X_name = "X"
        if "G_indptr" in self.outputs():
            cc.reshape(X_name, "X_BN3", shape=[self.batch_size(), -1, self.data_shape()[-1]])  # BxNx3
            X_name = "X_BN3"
        distance_matrix(cc, X_name, self.reconstructed_name(), "DistMat")
        self.distance_loss_func()(cc, "DistMat", "distance")
        cc.reduce("distance", "loss", axis=0, loss_weight=1, coeff=1.0 / B)

        cc.comment_blob_shape()
        if plot:
            cc.done(draw_net=ccdraw)
        else:
            cc.done()
        print("written:" + cc.filepath)


class BaseFoldingNetDecoder(BaseAutoEncoder):
    def generate_grid_and_replicate_code(self):
        raise NotImplementedError()

    def decoder(self):
        """Code -> Xp"""
        cc = self.cc
        cc.comment_bar("Decoder")
        grid, code_rep = self.generate_grid_and_replicate_code()

        cc.comment("First folding")
        cc.concat([grid, code_rep], "Y1", axis=-1)
        self.XKrelu("Y1", "Y2", 512).XKrelu("Y2", "Y3", 512).XK("Y3", "Y", 3).cc.space()

        cc.comment("Second folding")
        cc.concat(["Y", code_rep], "Z1", axis=-1)
        self.XKrelu("Z1", "Z2", 512).XKrelu("Z2", "Z3", 512).XK(
            "Z3",
            self.reconstructed_name(),
            3,
        ).cc.space()


class BaseFoldingNetDecoderND(BaseFoldingNetDecoder):
    def get_nd_grid_resolution(self):
        raise NotImplementedError()

    def generate_grid_and_replicate_code(self):
        cc = self.cc
        Ms = self.get_nd_grid_resolution()
        M = np.prod(Ms)
        cc.pydata(
            outputs=[
                "Grid",
            ],
            shapes=[
                [self.batch_size(), M, len(Ms)],
            ],
            module="io_layers",
            layer="GridSamplingLayer",
            param_str="{'batch_size':%d, 'meshgrid':%s}" % (self.batch_size(), str([[-0.3, 0.3, Mi] for Mi in Ms])),
            phase="",
        )  # BxMxlen(Ms)
        cc.reshape(self.code_name(), self.code_name() + "_B1K", shape=[0, 1, -1])
        cc.tile(self.code_name() + "_B1K", self.code_name() + "_BMK", axis=-2, tiles=M).space()  # BxMxK
        return "Grid", self.code_name() + "_BMK"


class FoldingNetDecoder2D(BaseFoldingNetDecoderND):
    def get_nd_grid_resolution(self):
        return [45, 45]


class FoldingNetDecoder3D(BaseFoldingNetDecoderND):
    def get_nd_grid_resolution(self):
        return [13, 13, 13]


class FoldingNetDecoder2D3(BaseFoldingNetDecoderND):
    def get_nd_grid_resolution(self):
        return [45, 45, 1]


class BaseFoldingNetRNNDecoder(BaseFoldingNetDecoder):
    def decoder(self):
        """Code -> Xp"""
        cc = self.cc
        cc.comment_bar("Decoder")
        grid, code_rep = self.generate_grid_and_replicate_code()
        grid_shape = cc.shape_of(grid)
        X_shape = cc.shape_of("X")
        assert grid_shape[-1] == X_shape[-1]  # otherwise we can not share weight

        cc.comment("First folding")
        cc.concat([grid, code_rep], "Y1", axis=-1)
        self.XKrelu("Y1", "Y2", 512, W_name="W[Y1=>Y2]", b_name="b[Y1=>Y2]").XKrelu(
            "Y2", "Y3", 512, W_name="W[Y2=>Y3]", b_name="b[Y2=>Y3]"
        ).XK("Y3", "Y", 3, W_name="W[Y3=>Y]", b_name="b[Y3=>Y]").cc.space()

        cc.comment("Second folding")
        cc.concat(["Y", code_rep], "Z1", axis=-1)
        self.XKrelu("Z1", "Z2", 512, W_name="W[Y1=>Y2]", b_name="b[Y1=>Y2]").XKrelu(
            "Z2", "Z3", 512, W_name="W[Y2=>Y3]", b_name="b[Y2=>Y3]"
        ).XK("Z3", self.reconstructed_name(), 3, W_name="W[Y3=>Y]", b_name="b[Y3=>Y]").cc.space()


class BaseLongFoldingNetRNNDecoder(BaseFoldingNetDecoder):
    def decoder(self):
        """Code -> Xp"""
        cc = self.cc
        cc.comment_bar("Decoder")
        grid, code_rep = self.generate_grid_and_replicate_code()
        grid_shape = cc.shape_of(grid)
        X_shape = cc.shape_of("X")
        assert grid_shape[-1] == X_shape[-1]  # otherwise we can not share weight

        cc.comment("First folding")
        cc.concat([grid, code_rep], "Y1", axis=-1)
        self.XKrelu("Y1", "Y2", 512, W_name="W[Y1=>Y2]", b_name="b[Y1=>Y2]").XKrelu(
            "Y2", "Y3", 512, W_name="W[Y2=>Y3]", b_name="b[Y2=>Y3]"
        ).XKrelu("Y3", "Y4", 256, W_name="W[Y3=>Y4]", b_name="b[Y3=>Y4]").XKrelu(
            "Y4", "Y5", 128, W_name="W[Y4=>Y5]", b_name="b[Y4=>Y5]"
        ).XK(
            "Y5", "Y", 3, W_name="W[Y5=>Y]", b_name="b[Y5=>Y]"
        ).cc.space()

        cc.comment("Second folding")
        cc.concat(["Y", code_rep], "Z1", axis=-1)
        self.XKrelu("Z1", "Z2", 512, W_name="W[Y1=>Y2]", b_name="b[Y1=>Y2]").XKrelu(
            "Z2", "Z3", 512, W_name="W[Y2=>Y3]", b_name="b[Y2=>Y3]"
        ).XKrelu("Z3", "Z4", 256, W_name="W[Y3=>Y4]", b_name="b[Y3=>Y4]").XKrelu(
            "Z4", "Z5", 128, W_name="W[Y4=>Y5]", b_name="b[Y4=>Y5]"
        ).XK(
            "Z5", self.reconstructed_name(), 3, W_name="W[Y5=>Y]", b_name="b[Y5=>Y]"
        ).cc.space()
