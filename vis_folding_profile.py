# Copyright (C) 2017-2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

import visualize as vis


def calc_4nn_maxdist(P, nh, nw, plot=False):
    jj, ii = np.meshgrid(range(nh), range(nw))
    iii = ii[1:-1, 1:-1]
    jjj = jj[1:-1, 1:-1]
    i = np.ravel_multi_index((iii.reshape(-1), jjj.reshape(-1)), dims=(nh, nw))
    il = i - 1
    ir = i + 1
    iu = i - nw
    id = i + nw

    dl = np.sqrt(np.sum((P[i, :] - P[il, :]) ** 2, axis=1))
    dr = np.sqrt(np.sum((P[i, :] - P[ir, :]) ** 2, axis=1))
    du = np.sqrt(np.sum((P[i, :] - P[iu, :]) ** 2, axis=1))
    dd = np.sqrt(np.sum((P[i, :] - P[id, :]) ** 2, axis=1))

    ret = np.zeros(ii.shape, dtype=np.float32) + 1
    ret[1:-1, 1:-1] = np.max(np.stack((dl, dr, du, dd)), axis=0).reshape(nh - 2, nw - 2)

    if plot:
        vis.plt.figure()
        vis.plt.imshow(ret, cmap="jet")
        vis.plt.colorbar()
        vis.plt.title("Tear/Stretch")
        vis.plt.gca().set_axis_off()
    return ret


def calc_4nn_cur(P, nh, nw, plot=False):
    sk = 1
    jj, ii = np.meshgrid(range(nh), range(nw))
    iii = ii[sk:-sk, sk:-sk]
    jjj = jj[sk:-sk, sk:-sk]
    i = np.ravel_multi_index((iii.reshape(-1), jjj.reshape(-1)), dims=(nh, nw))
    il = i - sk
    ir = i + sk
    iu = i - nw * sk
    id = i + nw * sk

    vr = P[ir, :] - P[i, :]
    vu = P[iu, :] - P[i, :]
    N = np.stack([np.cross(vr[k, :], vu[k, :]) for k in range(vr.shape[0])])
    Nn = np.sqrt(np.sum(N**2, axis=1))
    Nn = Nn[..., np.newaxis]
    N = N / Nn  # normalize
    No = np.zeros(P.shape)
    No[i, :] = N
    N = No

    dl = 1 - np.sum(N[i, :] * N[il, :], axis=1)
    dr = 1 - np.sum(N[i, :] * N[ir, :], axis=1)
    du = 1 - np.sum(N[i, :] * N[iu, :], axis=1)
    dd = 1 - np.sum(N[i, :] * N[id, :], axis=1)

    ret = np.zeros(ii.shape, dtype=np.float32) + 2
    ret[sk:-sk, sk:-sk] = np.max(np.stack((dl, dr, du, dd)), axis=0).reshape(nh - sk * 2, nw - sk * 2)

    if plot:
        # manual color the boundaries
        ret[0 : sk + 1, :] = 1.75  # red
        ret[-1 - sk :, :] = 0.25  # blue
        ret[:, 0 : sk + 1] = 1.0  # green
        ret[:, -1 - sk :] = 1.28  # yellow

        ax, _ = vis.draw_pts(P, ret.reshape(-1), "jet")
        iii = ii[0, :]
        jjj = jj[0, :]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i, 0], P[i, 1], P[i, 2], zdir="y", color="red", linewidth=2)

        iii = ii[-1, :]
        jjj = jj[-1, :]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i, 0], P[i, 1], P[i, 2], zdir="y", color="blue", linewidth=2)

        iii = ii[:, 0]
        jjj = jj[:, 0]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i, 0], P[i, 1], P[i, 2], zdir="y", color="green", linewidth=2)

        iii = ii[:, -1]
        jjj = jj[:, -1]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i, 0], P[i, 1], P[i, 2], zdir="y", color="yellow", linewidth=2)

        vis.plt.figure()
        vis.plt.imshow(ret, cmap="jet")
        vis.plt.colorbar()
        vis.plt.title("Fold")
        vis.plt.gca().set_axis_off()
    return ret


if __name__ == "__main__":
    data = np.load("experiments/Primitive_FoldingNet_noGMCov/dense_interp/interp_0_1.npz")
    vis.draw_pts(data["Xb"], None, "gray")
    X = data["Xpb"]
    md = calc_4nn_cur(X, 100, 100, plot=True)
    cd = calc_4nn_maxdist(X, 100, 100, True)
    vis.plt.show()
