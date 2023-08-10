import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.autograd import Variable


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A
        if -1 != x.get_device():
            A = A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = (
                self.conv_a[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Classifier(nn.Module):
    def __init__(self, num_class=60, scale_factor=5.0, temperature=[1.0, 2.0, 5.0]):
        super(Classifier, self).__init__()

        # action features
        self.ac_center = nn.Parameter(torch.zeros(num_class + 1, 256))
        nn.init.xavier_uniform_(self.ac_center)
        # foreground feature

        self.temperature = temperature
        self.scale_factor = scale_factor

    def forward(self, x):

        N = x.size(0)

        x_emb = reduce(x, "(n m) c t v -> n t c", "mean", n=N)

        norms_emb = F.normalize(x_emb, dim=2)
        norms_ac = F.normalize(self.ac_center)

        # generate foeground and action scores
        frm_scrs = (
            torch.einsum("ntd,cd->ntc", [norms_emb, norms_ac]) * self.scale_factor
        )

        # attention
        class_wise_atts = [F.softmax(frm_scrs * t, 1) for t in self.temperature]

        # multiple instance learning branch
        # temporal score aggregation
        mid_vid_scrs = [
            torch.einsum("ntc,ntc->nc", [frm_scrs, att]) for att in class_wise_atts
        ]
        mil_vid_scr = (
            torch.stack(mid_vid_scrs, -1).mean(-1) * 2.0
        )  # frm_scrs have been multiplied by the scale factor
        mil_vid_pred = F.sigmoid(mil_vid_scr)

        return mil_vid_pred, frm_scrs


class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        scale_factor=5.0,
        temperature=[1.0, 2.0, 5.0],
    ):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A, stride=2)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        bn_init(self.data_bn, 1)

        self.classifier_1 = Classifier(num_class, scale_factor, temperature)

        self.classifier_2 = Classifier(num_class, scale_factor, temperature)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = rearrange(x, "n c t v m -> n (m v c) t")
        # x = self.data_bn(x)
        x = rearrange(x, "n (m v c) t -> (n m) c t v", m=M, v=V, c=C)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        mil_vid_pred_1, frm_scrs_1 = self.classifier_1(x)

        mil_vid_pred_2, frm_scrs_2 = self.classifier_2(x.detach())

        # print (frm_scrs_1.size(), T)

        frm_scrs_1 = rearrange(frm_scrs_1, "n t c -> n c t")
        frm_scrs_1 = F.interpolate(
            frm_scrs_1, size=(T), mode="linear", align_corners=True
        )
        frm_scrs_1 = rearrange(frm_scrs_1, "n c t -> n t c")

        frm_scrs_2 = rearrange(frm_scrs_2, "n t c -> n c t")
        frm_scrs_2 = F.interpolate(
            frm_scrs_2, size=(T), mode="linear", align_corners=True
        )
        frm_scrs_2 = rearrange(frm_scrs_2, "n c t -> n t c")

        return mil_vid_pred_1, frm_scrs_1, mil_vid_pred_2, frm_scrs_2
