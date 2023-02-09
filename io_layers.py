# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

import caffe
import numpy as np
import pyxis
import scipy.sparse as sparse

from utils import rand_ortho_rotation_matrix, rand_rotation_matrix


def fastprint(str):
    print(str)
    sys.stdout.flush()


class BaseIOLayer(caffe.Layer):
    """base io class supporting batch and random rotation augmentation

    params = {
    'source':'data/modelNet_train.npy',
    'batch_size': 64
    'random_seed': -1 #seed for random replace some portion with noise points
    'rand_rotation': '' # ''|'ortho'|'full'
    'rand_online': False,
    'output_label': False
    }
    """

    def set_member(self, name, default=None):
        setattr(self, name, self.params[name] if default is None or self.params.has_key(name) else default)

    def get_batch_indices(self, start_id):
        return [i % self.n_all_data for i in xrange(start_id, start_id + self.batch_size)]

    def next_batch(self):
        """invoked in self.forward(bottom,top) to move to next batch"""
        self.ith = (self.ith + self.batch_size) % self.n_all_data

    def augment_data(self, xyz, cov=None):
        """invoked either online in self.reshape(bottom,top)

        xyz <BxNxD>: batched point clouds
        """
        if self.rand_rotation:  # make random rotation
            rand_rot_func = (
                rand_ortho_rotation_matrix if self.rand_rotation.lower() == "ortho" else rand_rotation_matrix
            )
            for kth in xrange(xyz.shape[0]):
                R = rand_rot_func()
                xyz[kth, ...] = np.dot(xyz[kth], R)
                if cov is not None:
                    M = np.kron(R.T, R.T).transpose()
                    cov[kth, ...] = np.dot(
                        cov[kth], M
                    )  # vec(Cy.T).T = vec(Cx.T).T * kron(R.T, R.T).T if Cy = R.T * Cx * R

    def setup(self, bottom, top):
        self.params = eval(self.param_str)

        self.set_member("source")
        assert os.path.exists(self.source)

        self.set_member("rand_rotation", "")
        self.set_member("random_seed", -1)
        self.set_member("rand_online", False)
        self.set_member("output_label", False)

        if self.random_seed > 0:
            np.random.seed(np.uint32(self.random_seed))
        if self.rand_rotation:
            assert self.random_seed > 0

        self.set_member("batch_size")
        assert self.batch_size > 0

        self.load_data()
        assert hasattr(self, "n_all_data")

        self.ith = 0
        self.old_ith = -1
        assert self.batch_size <= self.n_all_data

        # fastprint('n_all_data={}'.format(self.n_all_data))

    def reshape(self, bottom, top):
        if self.ith == self.old_ith:
            return
        if self.ith < self.old_ith:  # restarted
            new_idx = np.random.permutation(self.n_all_data)
            self.reshuffle(new_idx)
        self.old_ith = self.ith

        bids = self.get_batch_indices(self.ith)
        self.update_current_data(bids, top)

    def backward(self, top, propagate_down, bottom):
        pass

    def forward(self, bottom, top):
        raise NotImplementedError()

    def load_data(self):
        """invoked only once in self.setup(bottom,top)"""
        raise NotImplementedError()

    def reshuffle(self, new_idx):
        """invoked in self.reshape(bottom,top) after each epoch"""
        raise NotImplementedError()

    def update_current_data(self, bids, top):
        """invoked in self.reshape(bottom,top) to load current batch with bids"""
        raise NotImplementedError()


class InputXYZCovNPYLayer(BaseIOLayer):
    """
    InputXYZCovNPYLayer -> X, Cov[, label]

    X <BxNx3>:   batched 3D Points, each containing N points representing a object instance
    Cov <BxNx9>: local covariance matrices of each point's k-nn neighbors
    label <B>:   label for each object instance

    params = {
    'source':'data/modelNet_train.npy',
    'batch_size': 64
    'random_seed': -1 #seed for random replace some portion with noise points
    'rand_rotation': '' # ''|'ortho'|'full'
    'rand_online': False,
    'output_label': False
    }
    """

    def load_data(self):
        raw_data = np.load(self.source).item()
        assert isinstance(raw_data, dict)
        self.all_data = raw_data["data"]  # [BxNx3]
        self.all_cov = raw_data["cov"]  # [BxNx9]
        assert len(self.all_data.shape) == 3
        assert self.all_data.shape[0] == self.all_cov.shape[0]
        assert self.all_data.shape[-1] == 3
        self.n_all_data = self.all_data.shape[0]

        if self.output_label:
            self.all_label = raw_data["label"]

    def reshuffle(self, new_idx):
        self.all_data = self.all_data[new_idx]
        self.all_cov = self.all_cov[new_idx]
        if self.output_label:
            self.all_label = self.all_label[new_idx]

    def update_current_data(self, bids, top):
        if self.rand_online:
            self.xyz = np.array(self.all_data[bids])  # BxNx3
            self.cov = np.array(self.all_cov[bids])
            self.augment_data(self.xyz, cov=self.cov)
        else:
            self.xyz = self.all_data[bids]
            self.cov = self.all_cov[bids]

        if self.output_label:
            self.label = self.all_label[bids]

        assert self.xyz.shape[0] == self.batch_size
        if self.output_label:
            assert len(top) == 3
        else:
            assert len(top) == 2
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.cov.shape)
        if self.output_label:
            top[2].reshape(*self.label.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.cov
        if self.output_label:
            top[2].data[...] = self.label
        self.next_batch()


class InputXYZCovGraphNPYLayer(BaseIOLayer):
    """
    InputXYZCovGraphNPYLayer -> X, Cov, n_offset, G_indptr, G_indices[, G_data]

    X <(B*N)x3>: batched 3D Points, each containing N points representing a object instance
    Cov <(B*N)x9>: covariance matrices
    n_offset <B,>: end indices for each data instance in the batch
    G_indptr, G_indices, G_data: batched sparse G matrix (block diagonal), G_data might be omitted

    params = {
    'source':'data/modelNet_train.npy',
    'batch_size': 64
    'mode': 'M'
    'random_seed': -1 #seed for random replace some portion with noise points
    'rand_rotation': '' # ''|'ortho'|'full'
    'rand_online': False,
    }
    """

    def load_data(self):
        self.set_member("mode", "M")
        assert self.mode in ["M", "P"]

        # Graph data should be recomputed if X changes, thus num_noise_pts should always be 0
        raw_data = np.load(self.source).item()
        assert isinstance(raw_data, dict)
        self.all_data = raw_data["data"]  # [BxNx3]
        self.all_cov = raw_data["cov"]
        raw_graph = raw_data["graph"]
        self.all_graph = np.asarray([g[self.mode] for g in raw_graph])
        assert len(self.all_data.shape) == 3
        assert self.all_data.shape[0] == self.all_cov.shape[0]
        assert self.all_data.shape[-1] == 3
        assert len(self.all_graph) == self.all_data.shape[0]
        self.n_all_data = self.all_data.shape[0]

        if self.output_label:
            self.all_label = raw_data["label"]

    def reshuffle(self, new_idx):
        self.all_data = self.all_data[new_idx]
        self.all_cov = self.all_cov[new_idx]
        self.all_graph = self.all_graph[new_idx]
        if self.output_label:
            self.all_label = self.all_label[new_idx]

    def update_current_data(self, bids, top):
        if self.rand_online:
            self.xyz = np.array(self.all_data[bids])
            self.cov = np.array(self.all_cov[bids])
            self.augment_data(self.xyz, cov=self.cov)
        else:
            self.xyz = self.all_data[bids]
            self.cov = self.all_cov[bids]

        if self.output_label:
            self.label = self.all_label[bids]

        self.xyz = self.xyz.reshape((-1, self.xyz.shape[-1]))  # reshape to (B*N)x3
        self.cov = self.cov.reshape((-1, self.cov.shape[-1]))  # reshape to (B*N)x9
        self.n_offset = (np.arange(self.batch_size) + 1) * self.all_data.shape[1]
        self.G = sparse.block_diag(self.all_graph[bids], format="csr")

        assert self.xyz.shape[0] < 16777216  # check graph_pooling_layer Reshape() for more information
        assert self.G.indices.shape[0] < 16777216  # if assertion failed, use a smaller self.batch_size!!
        assert isinstance(self.G, sparse.csr_matrix)

        assert self.xyz.shape[0] == self.batch_size * self.all_data.shape[1]
        if self.output_label:
            assert len(top) in [6, 7]
        else:
            assert len(top) in [5, 6]

        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.cov.shape)
        top[2].reshape(*self.n_offset.shape)
        top[3].reshape(*self.G.indptr.shape)
        top[4].reshape(*self.G.indices.shape)
        if self.output_label:
            if len(top) == 6:
                top[5].reshape(*self.label.shape)
            else:
                top[5].reshape(*self.G.data.shape)
                top[6].reshape(*self.label.shape)
        else:
            if len(top) > 5:
                top[5].reshape(*self.G.data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.cov
        top[2].data[...] = self.n_offset
        top[3].data[...] = self.G.indptr
        top[4].data[...] = self.G.indices
        if self.output_label:
            if len(top) == 6:
                top[5].data[...] = self.label
            else:
                top[5].data[...] = self.G.data
                top[6].data[...] = self.label
        else:
            if len(top) > 5:
                top[5].data[...] = self.G.data
        self.next_batch()


class IOShapeNetCovLMDBLayer(BaseIOLayer):
    """
    IOShapeNetCovLMDBLayer -> X, Cov
    """

    def load_data(self):
        self.db = pyxis.Reader(dirpath=self.source)
        keys = self.db.get_data_keys()
        assert "data" in keys
        assert "cov" in keys
        self.n_all_data = self.db.nb_samples
        self.sample_idx = range(self.n_all_data)

    def reshuffle(self, new_idx):
        self.sample_idx = new_idx.tolist()
        # fastprint('reached end of dataset, reshuffle...')

    def update_current_data(self, bids, top):
        self.xyz = []
        self.cov = []
        for bid in bids:
            sample = self.db.get_sample(self.sample_idx[bid])
            self.xyz.append(sample["data"])
            self.cov.append(sample["cov"])
        self.xyz = np.stack(self.xyz)
        self.cov = np.stack(self.cov)

        if self.rand_online:
            self.augment_data(self.xyz, cov=self.cov)

        assert self.xyz.shape[0] == self.batch_size
        assert len(top) == 2
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.cov.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.cov
        self.next_batch()


class InputXYZCovGraphLMDBLayer(IOShapeNetCovLMDBLayer):
    """
    InputXYZCovGraphLMDBLayer -> X, Cov, n_offset, G_indptr, G_indices[, G_data]
    """

    def update_current_data(self, bids, top):
        self.xyz = []
        self.cov = []
        self.graph = []

        for bid in bids:
            sample = self.db.get_sample(self.sample_idx[bid])
            self.xyz.append(sample["data"])
            N = sample["data"].shape[-2]
            self.cov.append(sample["cov"])
            self.graph.append(
                sparse.csr_matrix((sample["G_data"], sample["G_indices"], sample["G_indptr"]), shape=(N, N))
            )

        self.xyz = np.stack(self.xyz)  # BxNx3
        self.cov = np.stack(self.cov)  # BxNx9
        if self.rand_online:
            self.augment_data(self.xyz, cov=self.cov)

        N = self.xyz[0].shape[-2]
        self.xyz = self.xyz.reshape(-1, 3)  # (B*N)x3
        self.cov = self.cov.reshape(-1, 9)  # (B*N)x9
        self.n_offset = (np.arange(self.batch_size) + 1) * N
        self.G = sparse.block_diag(self.graph, format="csr")

        assert self.xyz.shape[0] == self.batch_size * N
        assert len(top) in [5, 6]
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.cov.shape)
        top[2].reshape(*self.n_offset.shape)
        top[3].reshape(*self.G.indptr.shape)
        top[4].reshape(*self.G.indices.shape)
        if len(top) > 5:
            top[5].reshape(*self.G.data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.cov
        top[2].data[...] = self.n_offset
        top[3].data[...] = self.G.indptr
        top[4].data[...] = self.G.indices
        if len(top) > 5:
            top[5].data[...] = self.G.data
        self.next_batch()


class GridSamplingLayer(caffe.Layer):
    """
    output Grid points as a NxD matrix

    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    """

    def set_member(self, name, default=None):
        setattr(self, name, self.params[name] if default is None or self.params.has_key(name) else default)

    def setup(self, bottom, top):
        self.params = eval(self.param_str)
        self.set_member("batch_size")
        self.set_member("meshgrid")

        ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in self.meshgrid])
        ndim = len(self.meshgrid)
        grid = np.zeros((np.prod([it[2] for it in self.meshgrid]), ndim), dtype=np.float32)  # MxD
        for d in xrange(ndim):
            grid[:, d] = np.reshape(ret[d], -1)
        self.grid = np.repeat(grid[np.newaxis, ...], repeats=self.batch_size, axis=0)

    def reshape(self, bottom, top):
        top[0].reshape(*self.grid.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.grid

    def backward(self, top, propagate_down, bottom):
        pass
