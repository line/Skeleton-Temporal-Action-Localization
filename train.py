#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
# Adapted from https://github.com/lshiwjx/2s-AGCN for BABEL (https://babel.is.tue.mpg.de/)

from __future__ import print_function

import argparse
import inspect
import os
import pdb
import pickle
import random
import re
import shutil
import time
from collections import *

import ipdb
import numpy as np

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from einops import rearrange, reduce, repeat
from evaluation.classificationMAP import getClassificationMAP as cmAP
from evaluation.detectionMAP import getSingleStreamDetectionMAP as dsmAP
from feeders.tools import collate_with_padding_multi_joint
from model.losses import cross_entropy_loss, mvl_loss
from sklearn.metrics import f1_score

# Custom
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from utils.logger import Logger


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description="Spatial Temporal Graph Convolution Network"
    )
    parser.add_argument(
        "--work-dir",
        default="./work_dir/temp",
        help="the work folder for storing results",
    )

    parser.add_argument("-model_saved_name", default="")
    parser.add_argument(
        "--config",
        default="./config/nturgbd-cross-view/test_bone.yaml",
        help="path to the configuration file",
    )

    # processor
    parser.add_argument("--phase", default="train", help="must be train or test")

    # visulize and debug
    parser.add_argument("--seed", type=int, default=5, help="random seed for pytorch")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="the interval for printing messages (#iteration)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=2,
        help="the interval for storing models (#iteration)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="the interval for evaluating models (#iteration)",
    )
    parser.add_argument(
        "--print-log", type=str2bool, default=True, help="print logging or not"
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=[1, 5],
        nargs="+",
        help="which Top K accuracy will be shown",
    )

    # feeder
    parser.add_argument(
        "--feeder", default="feeder.feeder", help="data loader will be used"
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=32,
        help="the number of worker for data loader",
    )
    parser.add_argument(
        "--train-feeder-args",
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--test-feeder-args",
        default=dict(),
        help="the arguments of data loader for test",
    )

    # model
    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument(
        "--model-args", type=dict, default=dict(), help="the arguments of model"
    )
    parser.add_argument(
        "--weights", default=None, help="the weights for network initialization"
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )

    # optim
    parser.add_argument(
        "--base-lr", type=float, default=0.01, help="initial learning rate"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=[200],
        nargs="+",
        help="the epoch where optimizer reduce the learning rate",
    )

    # training
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument("--optimizer", default="SGD", help="type of optimizer")
    parser.add_argument(
        "--nesterov", type=str2bool, default=False, help="use nesterov or not"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="start training from which epoch"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=80, help="stop training in which epoch"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="weight decay for optimizer"
    )
    # loss
    parser.add_argument("--loss", type=str, default="CE", help="loss type(CE or focal)")
    parser.add_argument(
        "--label_count_path",
        default=None,
        type=str,
        help="Path to label counts (used in loss weighting)",
    )
    parser.add_argument(
        "---beta",
        type=float,
        default=0.9999,
        help="Hyperparameter for Class balanced loss",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="Hyperparameter for Focal loss"
    )

    parser.add_argument("--only_train_part", default=False)
    parser.add_argument("--only_train_epoch", default=0)
    parser.add_argument("--warm_up_epoch", default=0)

    parser.add_argument(
        "--lambda-mil", default=1.0, help="balancing hyper-parameter of mil branch"
    )

    parser.add_argument(
        "--class-threshold",
        type=float,
        default=0.1,
        help="class threshold for rejection",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.03,
        help="start threshold for action localization",
    )
    parser.add_argument(
        "--end-threshold",
        type=float,
        default=0.055,
        help="end threshold for action localization",
    )
    parser.add_argument(
        "--threshold-interval",
        type=float,
        default=0.005,
        help="threshold interval for action localization",
    )
    return parser


class Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == "train":
            if not arg.train_feeder_args["debug"]:
                if os.path.isdir(arg.model_saved_name):
                    print("log_dir: ", arg.model_saved_name, "already exist")
                    # answer = input('delete it? y/n:')
                    answer = "y"
                    if answer == "y":
                        print("Deleting dir...")
                        shutil.rmtree(arg.model_saved_name)
                        print("Dir removed: ", arg.model_saved_name)
                        # input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print("Dir not removed: ", arg.model_saved_name)
                self.train_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "train"), "train"
                )
                self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "val"), "val"
                )
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "test"), "test"
                )
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_per_class_acc = 0
        self.loss_nce = torch.nn.BCELoss()

        self.my_logger = Logger(
            os.path.join(arg.model_saved_name, "log.txt"), title="SWTAL"
        )
        self.my_logger.set_names(["Step", "cmap"] + [f"map_0.{i}" for i in range(1, 8)])

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                collate_fn=collate_with_padding_multi_joint,
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            collate_fn=collate_with_padding_multi_joint,
        )

    def load_model(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)
        self.loss_type = arg.loss

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split("-")[-1])
            self.print_log("Load weights from {}.".format(self.arg.weights))
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "r") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [
                    [k.split("module.")[-1], v.cuda(output_device)]
                    for k, v in weights.items()
                ]
            )

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(
                                "Sucessfully Remove Weights: {}.".format(key)
                            )
                        else:
                            self.print_log("Can Not Remove Weights: {}.".format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model, device_ids=self.arg.device, output_device=output_device
                )

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError()

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == "SGD" or self.arg.optimizer == "Adam":
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step))
                )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + " ] " + str
        print(str)
        if self.arg.print_log:
            with open("{}/print_log.txt".format(self.arg.work_dir), "a") as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, wb_dict, save_model=False):
        self.model.train()
        self.print_log("Training epoch: {}".format(epoch + 1))
        loader = self.data_loader["train"]
        self.adjust_learning_rate(epoch)

        loss_value, batch_acc = [], []
        self.train_writer.add_scalar("epoch", epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print("only train part, require grad")
                for key, value in self.model.named_parameters():
                    if "PA" in key:
                        value.requires_grad = True
            else:
                print("only train part, do not require grad")
                for key, value in self.model.named_parameters():
                    if "PA" in key:
                        value.requires_grad = False

        vid_preds = []
        frm_preds = []
        vid_lens = []
        labels = []

        results = []
        indexs = []

        for batch_idx, (data, label, target, mask, index, soft_label) in enumerate(
            process
        ):

            self.global_step += 1
            # get data
            data = data.float().cuda(self.output_device)
            label = label.cuda(self.output_device)
            mask = mask.cuda(self.output_device)
            soft_label = soft_label.cuda(self.output_device)
            timer["dataloader"] += self.split_time()

            indexs.extend(index.cpu().numpy().tolist())

            ab_labels = torch.cat([label, torch.ones(label.size(0), 1).cuda()], -1)

            # forward
            mil_pred, frm_scrs, mil_pred_2, frm_scrs_2 = self.model(data)

            cls_mil_loss = self.loss_nce(mil_pred, ab_labels.float()) + self.loss_nce(
                mil_pred_2, ab_labels.float()
            )

            if epoch > 10:

                frm_scrs_re = rearrange(frm_scrs, "n t c -> (n t) c")
                frm_scrs_2_re = rearrange(frm_scrs_2, "n t c -> (n t) c")
                soft_label = rearrange(soft_label, "n t c -> (n t) c")

                loss = cls_mil_loss * 0.1 + mvl_loss(
                    frm_scrs, frm_scrs_2, rate=0.2, weight=0.5
                )

                loss += cross_entropy_loss(
                    frm_scrs_re, soft_label
                ) + cross_entropy_loss(frm_scrs_2_re, soft_label)

            else:
                loss = cls_mil_loss * self.arg.lambda_mil + mvl_loss(
                    frm_scrs, frm_scrs_2, rate=0.2, weight=0.5
                )

            for i in range(data.size(0)):
                frm_scr = frm_scrs[i]

                label_ = label[i].cpu().numpy()
                mask_ = mask[i].cpu().numpy()
                vid_len = mask_.sum()

                frm_pred = F.softmax(frm_scr, -1).detach().cpu().numpy()[:vid_len]
                vid_pred = mil_pred[i].detach().cpu().numpy()

                results.append(frm_pred)

                vid_preds.append(vid_pred)
                frm_preds.append(frm_pred)
                vid_lens.append(vid_len)
                labels.append(label_)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer["model"] += self.split_time()

        vid_preds = np.array(vid_preds)
        frm_preds = np.array(frm_preds)
        vid_lens = np.array(vid_lens)
        labels = np.array(labels)

        loader.dataset.label_update(results, indexs)

        cmap = cmAP(vid_preds, labels)

        self.train_writer.add_scalar("acc", cmap, self.global_step)
        self.train_writer.add_scalar("loss", np.mean(loss_value), self.global_step)

        # statistics
        self.lr = self.optimizer.param_groups[0]["lr"]
        self.train_writer.add_scalar("lr", self.lr, self.global_step)
        timer["statistics"] += self.split_time()

        # statistics of time consumption and loss
        self.print_log("\tMean training loss: {:.4f}.".format(np.mean(loss_value)))
        self.print_log("\tAcc score: {:.3f}%".format(cmap))

        # Log
        wb_dict["train loss"] = np.mean(loss_value)
        wb_dict["train acc"] = cmap

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict(
                [[k.split("module.")[-1], v.cpu()] for k, v in state_dict.items()]
            )

            torch.save(
                weights,
                self.arg.model_saved_name + str(epoch) + ".pt",
            )

        return wb_dict

    @torch.no_grad()
    def eval(
        self,
        epoch,
        wb_dict,
        loader_name=["test"],
    ):
        self.model.eval()
        self.print_log("Eval epoch: {}".format(epoch + 1))

        vid_preds = []
        frm_preds = []
        vid_lens = []
        labels = []

        for ln in loader_name:
            loss_value = []
            step = 0
            process = tqdm(self.data_loader[ln])

            for batch_idx, (data, label, target, mask, index, soft_label) in enumerate(
                process
            ):
                data = data.float().cuda(self.output_device)
                label = label.cuda(self.output_device)
                mask = mask.cuda(self.output_device)

                ab_labels = torch.cat([label, torch.ones(label.size(0), 1).cuda()], -1)

                # forward
                mil_pred, frm_scrs, mil_pred_2, frm_scrs_2 = self.model(data)

                cls_mil_loss = self.loss_nce(
                    mil_pred, ab_labels.float()
                ) + self.loss_nce(mil_pred_2, ab_labels.float())

                loss_co = mvl_loss(frm_scrs, frm_scrs_2, rate=0.2, weight=0.5)

                loss = cls_mil_loss * self.arg.lambda_mil + loss_co

                loss_value.append(loss.data.item())

                for i in range(data.size(0)):
                    frm_scr = frm_scrs[i]
                    vid_pred = mil_pred[i]

                    label_ = label[i].cpu().numpy()
                    mask_ = mask[i].cpu().numpy()
                    vid_len = mask_.sum()

                    frm_pred = F.softmax(frm_scr, -1).cpu().numpy()[:vid_len]
                    vid_pred = vid_pred.cpu().numpy()

                    vid_preds.append(vid_pred)
                    frm_preds.append(frm_pred)
                    vid_lens.append(vid_len)
                    labels.append(label_)

                step += 1

            vid_preds = np.array(vid_preds)
            frm_preds = np.array(frm_preds)
            vid_lens = np.array(vid_lens)
            labels = np.array(labels)

            cmap = cmAP(vid_preds, labels)

            score = cmap
            loss = np.mean(loss_value)

            dmap, iou = dsmAP(
                vid_preds,
                frm_preds,
                vid_lens,
                self.arg.test_feeder_args["data_path"],
                self.arg,
                multi=True,
            )

            print("Classification map %f" % cmap)
            for item in list(zip(iou, dmap)):
                print("Detection map @ %f = %f" % (item[0], item[1]))

            self.my_logger.append([epoch + 1, cmap] + dmap)

            wb_dict["val loss"] = loss
            wb_dict["val acc"] = score

            if score > self.best_acc:
                self.best_acc = score

            print("Acc score: ", score, " model: ", self.arg.model_saved_name)
            if self.arg.phase == "train":
                self.val_writer.add_scalar("loss", loss, self.global_step)
                self.val_writer.add_scalar("acc", score, self.global_step)

            self.print_log(
                "\tMean {} loss of {} batches: {}.".format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)
                )
            )
            self.print_log("\tAcc score: {:.3f}%".format(score))

        return wb_dict

    def start(self):
        wb_dict = {}
        if self.arg.phase == "train":
            self.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            self.global_step = (
                self.arg.start_epoch
                * len(self.data_loader["train"])
                / self.arg.batch_size
            )

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                )
                wb_dict = {"lr": self.lr}

                # Train
                wb_dict = self.train(epoch, wb_dict, save_model=save_model)

                # Eval. on val set
                wb_dict = self.eval(epoch, wb_dict, loader_name=["test"])
                # Log stats. for this epoch
                print("Epoch: {0}\nMetrics: {1}".format(epoch, wb_dict))

            print(
                "best accuracy: ",
                self.best_acc,
                " model_name: ",
                self.arg.model_saved_name,
            )

        elif self.arg.phase == "test":
            if not self.arg.test_feeder_args["debug"]:
                wf = self.arg.model_saved_name + "_wrong.txt"
                rf = self.arg.model_saved_name + "_right.txt"
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.arg.print_log = False
            self.print_log("Model:   {}.".format(self.arg.model))
            self.print_log("Weights: {}.".format(self.arg.weights))

            wb_dict = self.eval(
                epoch=0,
                wb_dict=wb_dict,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
            )
            print("Inference metrics: ", wb_dict)
            self.print_log("Done.\n")


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    print("BABEL Action Recognition")
    print("Config: ", arg)
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
