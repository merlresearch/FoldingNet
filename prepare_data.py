# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import errno
import multiprocessing as multiproc
import os
import sys
from copy import deepcopy
from functools import partial

import gdown  # https://github.com/wkentaro/gdown
import glog as logger
import h5py
import numpy as np
import pyxis
import scipy.sparse
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def safe_makedirs(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass


def get_modelnet40_data_npy_name():
    train_npy = os.path.join(DATA_DIR, "modelNet40_train_file.npy")
    test_npy = os.path.join(DATA_DIR, "modelNet40_test_file.npy")
    return train_npy, test_npy


def download_modelnet40_data(num_points):
    assert 0 <= num_points <= 2048

    def h5_to_npy(h5files, npy_fname, num_points):
        all_data = []
        all_label = []
        for h5 in h5files:
            f = h5py.File(h5)
            data = f["data"][:]
            data = data[:, :num_points, :]
            label = f["label"][:]
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)  # <BxNx3>
        all_label = np.concatenate(all_label, axis=0)[:, 0]  # <B,>
        np.save(npy_fname, {"data": all_data, "label": all_label})
        logger.info("saved: " + npy_fname)

    train_npy, test_npy = get_modelnet40_data_npy_name()
    remote_name = "modelnet40_ply_hdf5_2048"

    if not os.path.exists(train_npy):
        www = "https://shapenet.cs.stanford.edu/media/{}.zip".format(remote_name)
        zipfile = os.path.basename(www)
        os.system("wget %s; unzip %s" % (www, zipfile))
        os.system("mv %s %s" % (zipfile[:-4], DATA_DIR))
        os.system("rm %s" % (zipfile))

        h5folder = os.path.join(DATA_DIR, remote_name)
        h5files = [f for f in os.listdir(h5folder) if f.endswith(".h5")]
        test_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith("ply_data_test")])
        train_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith("ply_data_train")])
        h5_to_npy(h5files=test_files, npy_fname=test_npy, num_points=num_points)
        h5_to_npy(h5files=train_files, npy_fname=train_npy, num_points=num_points)

        os.system("rm -r %s" % h5folder)

    td = np.load(train_npy).item()
    assert isinstance(td, dict)
    assert td.has_key("data")
    assert td.has_key("label")
    assert td["data"].shape == (9840, num_points, 3)
    assert td["label"].shape == (9840,)

    td = np.load(test_npy).item()
    assert isinstance(td, dict)
    assert td.has_key("data")
    assert td.has_key("label")
    assert td["data"].shape == (2468, num_points, 3)
    assert td["label"].shape == (2468,)


def get_shapenet_part_data_npy_name():
    train_npy = os.path.join(DATA_DIR, "shapenet_part_train_file.npy")
    test_npy = os.path.join(DATA_DIR, "shapenet_part_test_file.npy")
    return train_npy, test_npy


def download_shapenet_part_data(num_points):
    assert 0 <= num_points <= 2048

    def h5_to_npy(h5files, npy_fname, num_points):
        all_data = []
        all_label = []
        all_seglabel = []
        for h5 in h5files:
            f = h5py.File(h5)
            data = f["data"][:]
            data = data[:, :num_points, :]
            label = f["label"][:]
            seg = f["pid"][:]
            seg = seg[:, :num_points]
            all_data.append(data)
            all_label.append(label)
            all_seglabel.append(seg)
        all_data = np.concatenate(all_data, axis=0)  # <BxNx3>
        all_label = np.concatenate(all_label, axis=0)[:, 0]  # <B,>
        all_seglabel = np.concatenate(all_seglabel, axis=0)  # <BxN>
        np.save(npy_fname, {"data": all_data, "label": all_label, "seg_label": all_seglabel})
        logger.info("saved: " + npy_fname)

    train_npy, test_npy = get_shapenet_part_data_npy_name()

    if not os.path.exists(train_npy):
        www = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
        zipfile = os.path.basename(www)
        os.system("wget %s; unzip %s" % (www, zipfile))
        os.system("mv hdf5_data %s" % (DATA_DIR))
        os.system("rm %s" % (zipfile))

        h5folder = os.path.join(DATA_DIR, "hdf5_data")
        h5files = [f for f in os.listdir(h5folder) if f.endswith(".h5")]
        test_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith("ply_data_test")])
        val_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith("ply_data_val")])
        train_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith("ply_data_train")])
        train_files += val_files  # note: following pointnet++'s protocal
        h5_to_npy(h5files=test_files, npy_fname=test_npy, num_points=num_points)
        h5_to_npy(h5files=train_files, npy_fname=train_npy, num_points=num_points)

        os.system("rm -r %s" % h5folder)

    td = np.load(train_npy).item()
    assert isinstance(td, dict)
    assert td.has_key("data")
    assert td.has_key("label")
    assert td.has_key("seg_label")
    assert td["data"].shape == (14007, num_points, 3)
    assert td["label"].shape == (14007,)
    assert td["seg_label"].shape == (14007, num_points)

    td = np.load(test_npy).item()
    assert isinstance(td, dict)
    assert td.has_key("data")
    assert td.has_key("label")
    assert td.has_key("seg_label")
    assert td["data"].shape == (2874, num_points, 3)
    assert td["label"].shape == (2874,)
    assert td["seg_label"].shape == (2874, num_points)


def get_shapenet_data_npy_name():
    train_npy = os.path.join(DATA_DIR, "shapenet57448xyzonly.npz")
    return train_npy


def download_shapenet_data(num_points):
    assert 0 <= num_points <= 2048
    train_npy = get_shapenet_data_npy_name()

    download_file_from_google_drive_if_not_exist("1sJd5bdCg9eOo3-FYtchUVlwDgpVdsbXB", train_npy)

    td = dict(np.load(train_npy))
    assert td.has_key("data")
    assert td["data"].shape == (57448, num_points, 3)


##########################################################################################


def edges2A(edges, n_nodes, mode="P", sparse_mat_type=scipy.sparse.csr_matrix):
    """
    note: assume no (i,i)-like edge
    edges: <2xE>
    """
    edges = np.array(edges).astype(int)

    data_D = np.zeros(n_nodes, dtype=np.float32)
    for d in xrange(n_nodes):
        data_D[d] = len(np.where(edges[0] == d)[0])

    if mode.upper() == "M":  # 'M' means max pooling, which use the same graph matrix as the adjacency matrix
        data = np.ones(edges[0].shape[0], dtype=np.int32)
    elif mode.upper() == "P":
        data = 1.0 / data_D[edges[0]]
    else:
        raise NotImplementedError("edges2A with unknown mode=" + mode)

    return sparse_mat_type((data, edges), shape=(n_nodes, n_nodes))


def knn_search(data, knn, metric="euclidean", symmetric=True):
    """
    Args:
      data: Nx3
      knn: default=16
    """
    assert knn > 0
    n_data_i = data.shape[0]
    kdt = KDTree(data, leaf_size=30, metric=metric)

    nbs = kdt.query(data, k=knn + 1, return_distance=True)
    cov = np.zeros((n_data_i, 9), dtype=np.float32)
    adjdict = dict()
    # wadj = np.zeros((n_data_i, n_data_i), dtype=np.float32)
    for i in xrange(n_data_i):
        # nbsd = nbs[0][i]
        nbsi = nbs[1][i]
        cov[i] = np.cov(data[nbsi[1:]].T).reshape(-1)  # compute local covariance matrix
        for j in xrange(knn):
            if symmetric:
                adjdict[(i, nbsi[j + 1])] = 1
                adjdict[(nbsi[j + 1], i)] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
                # wadj[nbsi[j + 1], i] = 1.0 / nbsd[j + 1]
            else:
                adjdict[(i, nbsi[j + 1])] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
    edges = np.array(adjdict.keys(), dtype=int).T
    return edges, nbs[0], cov  # , wadj


def build_graph_core(ith_datai, args):
    try:
        ith, xyi = ith_datai  # xyi: 2048x3
        n_data_i = xyi.shape[0]
        edges, nbsd, cov = knn_search(xyi, knn=args.knn, metric=args.metric)
        ith_graph = edges2A(edges, n_data_i, args.mode, sparse_mat_type=scipy.sparse.csr_matrix)
        nbsd = np.asarray(nbsd)[:, 1:]
        nbsd = np.reshape(nbsd, -1)

        if ith % 500 == 0:
            logger.info("{} processed: {}".format(args.flag, ith))

        return ith, ith_graph, nbsd, cov
    except KeyboardInterrupt:
        exit(-1)


def build_graphs(data_dict, args):
    """
    Build graphs based on mode of all data.
    Data and graphs are saved in args.src (path).
    """
    total_num_data = data_dict["data"].shape[0]

    if args.shuffle:
        idx = np.arange(total_num_data)
        np.random.shuffle(idx)
        data_dict["data"] = data_dict["data"][idx]
        if data_dict.has_key("label"):
            data_dict["label"] = data_dict["label"][idx]
        if data_dict.has_key("seg_label"):
            data_dict["seg_label"] = data_dict["seg_label"][idx]

    graphs = [{} for i in range(total_num_data)]
    covs = np.zeros((total_num_data, data_dict["data"].shape[1], 9), dtype=np.float32)
    all_nbs_dist = []

    # parallel version
    pool = multiproc.Pool(multiproc.cpu_count())
    pool_func = partial(build_graph_core, args=args)
    rets = pool.map(pool_func, zip(range(total_num_data), data_dict["data"]))
    pool.close()
    for ret in rets:
        ith, ith_graph, nbsd, cov = ret
        graphs[ith][args.mode] = ith_graph
        covs[ith] = cov
        all_nbs_dist.append(nbsd)
    del rets

    all_nbs_dist = np.stack(all_nbs_dist)
    mean_nbs_dist = all_nbs_dist.mean()
    std_nbs_dist = all_nbs_dist.std()

    logger.info("{}: neighbor distance: mean={:f}, std={:f}".format(args.flag, mean_nbs_dist, std_nbs_dist))
    data_dict.update({"graph": graphs, "cov": covs, "mean_nbs_dist": mean_nbs_dist, "std_nbs_dist": std_nbs_dist})
    np.save(args.dst, data_dict)
    logger.info("saved: " + args.dst)


def build_graphs_lmdb(data_dict, args, max_num_data_per_section=1000):
    """
    Build graphs based on mode of all data.
    Data and graphs are saved in args.src (path).
    """
    total_num_data = data_dict["data"].shape[0]

    if args.shuffle:
        idx = np.arange(total_num_data)
        np.random.shuffle(idx)
        data_dict["data"] = data_dict["data"][idx]

    pool = multiproc.Pool(multiproc.cpu_count())
    pool_func = partial(build_graph_core, args=args)
    db = pyxis.Writer(dirpath=args.dst, map_size_limit=50000)

    n_section = int(np.ceil(float(total_num_data) / max_num_data_per_section))
    logger.info("{} number of sections: {}".format(args.flag, n_section))
    for sec_i in range(n_section):
        sec_ids = range(sec_i * max_num_data_per_section, min((sec_i + 1) * max_num_data_per_section, total_num_data))
        sec_data = data_dict["data"][sec_ids]

        rets = pool.map(pool_func, zip(sec_ids, sec_data))

        for ret in rets:
            ith, ith_graph, _, ith_cov = ret
            ith_data = data_dict["data"][ith]
            db.put_samples(
                {
                    "data": ith_data[np.newaxis, ...],
                    "cov": ith_cov[np.newaxis, ...],
                    "G_indices": ith_graph.indices[np.newaxis, ...],
                    "G_indptr": ith_graph.indptr[np.newaxis, ...],
                    "G_data": ith_graph.data[np.newaxis, ...],
                }
            )

        logger.info("{} saved section {}, length={}".format(args.flag, sec_i, len(rets)))

    db.close()
    pool.close()

    logger.info("saved: " + args.dst)


##########################################################################################


def load_data(data_file):
    if data_file.endswith(".npy"):
        data_dict = np.load(data_file).item()
    elif data_file.endswith(".npz"):
        data_dict = dict(np.load(data_file))
    else:
        raise IOError("unknown file format: " + data_file)
    assert isinstance(data_dict, dict)
    total_data = data_dict["data"]
    logger.info("all data size:")
    logger.info("data: " + str(total_data.shape))
    if data_dict.has_key("label"):
        logger.info("label: " + str(data_dict["label"].shape))
    if data_dict.has_key("seg_label"):
        logger.info("seg_label: " + str(data_dict["seg_label"].shape))

    concat = np.concatenate(total_data, axis=0)
    flag = os.path.basename(data_file)
    logger.info("{}: all data points locates within:".format(flag))
    logger.info("{}: min: {}".format(flag, tuple(concat.min(axis=0))))
    logger.info("{}: max: {}".format(flag, tuple(concat.max(axis=0))))

    return data_dict


def raw_npy_to_graph_and_cov_npy(raw_npy, args):
    fname_prefix = os.path.splitext(raw_npy)[0]
    return "{}_{}nn_G{}_cov.npy".format(fname_prefix, args.knn, args.mode)


def process_one_npy(args):
    args.flag = os.path.basename(args.src)
    args.dst = raw_npy_to_graph_and_cov_npy(args.src, args)
    if os.path.exists(args.dst):
        logger.info("{} existed already.".format(args.dst))
        return
    logger.info("{} ==(build_graphs)==> {}".format(args.src, args.dst))
    data_dict = load_data(args.src)
    build_graphs(data_dict, args)


def run_all_processes(all_p):
    try:
        for p in all_p:
            p.start()
        for p in all_p:
            p.join()
    except KeyboardInterrupt:
        for p in all_p:
            if p.is_alive():
                p.terminate()
            p.join()
        exit(-1)


def process_all_npy(all_npy, args):
    all_p = []
    for the_npy in all_npy:
        the_args = deepcopy(args)
        the_args.src = the_npy
        p = multiproc.Process(target=process_one_npy, args=(the_args,))
        all_p.append(p)
    run_all_processes(all_p)


##########################################################################################


def download_foldingnet_experiments_from_google_drive_and_unzip():
    www = "https://drive.google.com/uc?id=1aBA-XYZki3MhjgqXPSi3KyopVZFqCZVo"
    zipfile = "foldingnet_experiments.tar.gz"
    gdown.download(url=www, output=zipfile, quiet=False)
    os.system("tar xvzf %s -C %s" % (zipfile, BASE_DIR))
    os.system("rm %s" % (zipfile))


def download_file_from_google_drive_if_not_exist(ids, fname):
    if not os.path.exists(fname):
        gdown.download(url="https://drive.google.com/uc?id={:s}".format(ids), output=fname, quiet=False)
        assert os.path.exists(fname)
    else:
        logger.info("{} data file exited!".format(fname))


def download_cov_npy_from_google_drive(all_ids, npys, args):
    if not isinstance(npys, list):
        npys = [
            npys,
        ]
    if not isinstance(all_ids, list):
        all_ids = [
            all_ids,
        ]
    for ids, npy in zip(all_ids, npys):
        cov_npy = raw_npy_to_graph_and_cov_npy(npy, args)
        download_file_from_google_drive_if_not_exist(ids, cov_npy)


def prepare_modelnet40(args):
    logger.info("preparing modelnet40")
    train_npy, test_npy = get_modelnet40_data_npy_name()
    args.knn = 16

    if args.pts_mn40 == 2048 and args.mode == "M" and args.knn == 16 and not args.regenerate:
        download_cov_npy_from_google_drive(
            all_ids=["1XwF5bYofnx6_BjFXeMJvHVq98_t3eFjx", "19iNsy0PV8dn69d6ppda7tGP9PNoAsY_V"],
            npys=[train_npy, test_npy],
            args=args,
        )
    else:
        download_modelnet40_data(num_points=args.pts_mn40)
        process_all_npy([train_npy, test_npy], args=args)
    logger.info("modelnet40 done!")


def prepare_shapenet_part(args):
    logger.info("preparing shapenet_part")
    train_npy, test_npy = get_shapenet_part_data_npy_name()
    args.knn = 16

    if args.pts_shapenet_part == 2048 and args.mode == "M" and args.knn == 16 and not args.regenerate:
        download_cov_npy_from_google_drive(
            all_ids=["1QwdFeKQIgEgWgPRhaSCo2UTDpPx5jrJY", "1WMnAgwIDH8JPdrk89kvIndUtMU4SxfWs"],
            npys=[train_npy, test_npy],
            args=args,
        )
    else:
        download_shapenet_part_data(num_points=args.pts_shapenet_part)
        process_all_npy([train_npy, test_npy], args=args)
    logger.info("shapenet_part done!")


def prepare_shapenet(args):
    logger.info(
        "preparing shapenet: this dataset is huge after preparation (22.5GB), so we generate instead of download!"
    )
    train_npy = get_shapenet_data_npy_name()
    args.knn = 16

    download_shapenet_data(num_points=args.pts_shapenet)
    args = deepcopy(args)
    args.src = train_npy
    args.flag = os.path.basename(args.src)
    fname_prefix = os.path.splitext(args.src)[0]
    args.dst = "{}_{}nn_G{}_cov.lmdb".format(fname_prefix, args.knn, args.mode)
    if os.path.exists(args.dst):
        logger.info("{} existed already.".format(args.dst))
        return
    logger.info("{} ==(build_graphs)==> {}".format(args.src, args.dst))
    data_dict = load_data(args.src)
    build_graphs_lmdb(data_dict, args)
    logger.info("shapenet done!")


def main(args):
    assert len(args.mode) == 1
    safe_makedirs(DATA_DIR)
    all_p = []
    all_p.append(multiproc.Process(target=prepare_modelnet40, args=(deepcopy(args),)))
    all_p.append(multiproc.Process(target=prepare_shapenet_part, args=(deepcopy(args),)))
    all_p.append(multiproc.Process(target=prepare_shapenet, args=(deepcopy(args),)))
    all_p.append(multiproc.Process(target=download_foldingnet_experiments_from_google_drive_and_unzip))
    run_all_processes(all_p)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument("--pts_mn40", type=int, default=2048, help="number of points per modelNet40 object")
    parser.add_argument("--pts_shapenet_part", type=int, default=2048, help="number of points per shapenet_part object")
    parser.add_argument("--pts_shapenet", type=int, default=2048, help="number of points per shapenet object")
    parser.add_argument("-md", "--mode", type=str, default="M", help="mode used to compute graphs: M, P")
    parser.add_argument(
        "-m", "--metric", type=str, default="euclidean", help="metric for distance calculation (manhattan/euclidean)"
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        default=True,
        help="whether to shuffle data (1) or not (0) before saving",
    )
    parser.add_argument(
        "--regenerate",
        dest="regenerate",
        action="store_true",
        default=False,
        help="regenerate from raw pointnet data or not (default: False)",
    )

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
