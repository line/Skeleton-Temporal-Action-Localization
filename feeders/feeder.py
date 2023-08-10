#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
# Adapted from https://github.com/lshiwjx/2s-AGCN for BABEL (https://babel.is.tue.mpg.de/)

import json
import math
import os
import os.path as osp
import pdb
import pickle
import random
import shutil
import subprocess
import sys
import uuid

import matplotlib.pyplot as plt
import numpy as np
import torch
from feeders import tools
from torch.utils.data import Dataset

sys.path.extend(["../"])


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        debug=False,
        use_mmap=True,
        frame_pad=False,
        nb_class=3,
    ):
        """

        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.use_mmap = use_mmap
        self.nb_class = nb_class
        self.frame_pad = frame_pad
        self.load_data()
        self.count = 0
        for i in range(len(self.data["X"])):
            assert self.data["L"][i].shape[0] == self.data["X"][i].shape[1]

        self.prediction = [
            np.zeros((item.shape[0], 10, self.nb_class + 1), dtype=np.float32)
            for item in self.data["L"]
        ]
        self.soft_labels = [
            np.zeros((item.shape[0], self.nb_class + 1), dtype=np.float32)
            for item in self.data["L"]
        ]

    def load_data(self):
        # data: N, C, T, V, M
        # load data
        try:
            with open(self.data_path) as f:
                self.data = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f, encoding="latin1")

    def label_update(self, results, indexs):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10

        for ind, res in zip(indexs, results):
            self.prediction[ind][:, idx, :] = res

        for i in range(len(self.prediction)):
            self.soft_labels[i] = self.prediction[i].mean(axis=1)

    def __len__(self):
        return len(self.data["X"])

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data["X"][index]
        data_numpy = np.array(data_numpy)

        label = self.data["Y"][index]
        label_np = np.zeros(self.nb_class)
        for item in label:
            label_np[item] = 1
        label = np.array(label_np)

        gt = self.data["L"][index]
        gt = np.array(gt)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        if self.frame_pad:
            C, T, V, M = data_numpy.shape
            if T % 15 != 0:
                new_T = T + 15 - T % 15

                data_numpy_paded = np.zeros((C, new_T, V, M))
                data_numpy_paded[:, :T, :, :] = data_numpy

                data_numpy = data_numpy_paded

        mask = np.ones_like(gt)

        frame_label = self.soft_labels[index]

        return data_numpy, label, gt, mask, index, frame_label


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(
    dataset,
    preds=None,
    th=None,
    idx=None,
    graph="graph.ntu_rgb_d.Graph",
    is_3d=True,
    folder_p="viz",
    label_json="../../action_recognition/data/action_label_2_idx_3.json",
):
    """
    vis the samples using matplotlib
    :param data_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    """

    with open(label_json) as infile:
        jc = json.load(infile)

    idx2act = {v: k for k, v in jc.items()}

    idx2act[len(idx2act)] = "other"

    if osp.exists(osp.join(folder_p, "frames")):
        shutil.rmtree(osp.join(folder_p, "frames"))
    os.makedirs(osp.join(folder_p, "frames"))

    data, label, gt, _ = dataset[idx]
    data = data.reshape((1,) + data.shape)

    # for batch_idx, (data, label) in enumerate(loader):
    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    if graph is None:
        p_type = ["b.", "g.", "r.", "c.", "m.", "y.", "k.", "k.", "k.", "k."]
        pose = [ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)]
        ax.axis([-1, 1, -1, 1])
        for t in range(T):
            for m in range(M):
                pose[m].set_xdata(data[0, 0, t, :, m])
                pose[m].set_ydata(data[0, 1, t, :, m])
            fig.canvas.draw()
            plt.pause(0.001)
    else:
        p_type = ["b-", "g-", "r-", "c-", "m-", "y-", "k-", "k-", "k-", "k-"]
        import sys
        from os import path

        sys.path.append(
            path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
        )
        G = import_class(graph)()
        edge = G.inward
        pose = []
        for m in range(M):
            a = []
            for i in range(len(edge)):
                if is_3d:
                    a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                else:
                    a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
            pose.append(a)
        ax.axis([-1, 1, -1, 1])
        if is_3d:
            ax.set_zlim3d(-1, 1)
        for t in range(T):
            for m in range(M):
                for i, (v1, v2) in enumerate(edge):
                    x1 = data[0, :2, t, v1, m]
                    x2 = data[0, :2, t, v2, m]
                    if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                        pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                        pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                        if is_3d:
                            pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])

            if gt[t]:
                text = ax.text2D(
                    0.1, 0.9, idx2act[int(label)], size=20, transform=ax.transAxes
                )

            if preds is not None:
                pred_idx = preds[t].argmax()
                text_pred = ax.text2D(
                    0.6,
                    0.9,
                    idx2act[int(pred_idx)] + f": {preds[t, pred_idx]:.2f}",
                    size=20,
                    transform=ax.transAxes,
                )

            fig.canvas.draw()
            plt.savefig(osp.join(folder_p, "frames", str(t) + ".jpg"), dpi=300)
            if gt[t]:
                text.remove()
            if preds is not None:
                text_pred.remove()

        write_vid_from_imgs(folder_p, idx)


def write_vid_from_imgs(folder_p, fname, fps=30):
    """Collate frames into a video sequence.

    Args:
        folder_p (str): Frame images are in the path: folder_p/frames/<int>.jpg
        fps (float): Output frame rate.

    Returns:
        Output video is stored in the path: folder_p/video.mp4
    """
    vid_p = osp.join(folder_p, f"{fname}.mp4")
    cmd = [
        "ffmpeg",
        "-r",
        str(int(fps)),
        "-i",
        osp.join(folder_p, "frames", "%d.jpg"),
        "-y",
        vid_p,
    ]
    FNULL = open(os.devnull, "w")
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    if not 0 == retcode:
        print(
            "*******ValueError(Error {0} executing command: {1}*********".format(
                retcode, " ".join(cmd)
            )
        )
    shutil.rmtree(osp.join(folder_p, "frames"))


if __name__ == "__main__":
    import os

    os.environ["DISPLAY"] = "localhost:10.0"
    data_path = "../../data/train_loc_3.pkl"
    graph = "graph.ntu_rgb_d.Graph"
    dataset = Feeder(data_path)
    test(dataset, idx=0, graph=graph, is_3d=True)
