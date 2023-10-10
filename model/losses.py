"""
Copyright 2023 LINE Corporation
LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

from platform import mac_ver

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.autograd import Variable


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(
        F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False
    )
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def mvl_loss(y_1, y_2, rate=0.2, weight=0.1):
    y_1 = rearrange(y_1, "n t c -> (n t) c")
    y_2 = rearrange(y_2, "n t c -> (n t) c")

    loss_pick = weight * kl_loss_compute(
        y_1, y_2, reduce=False
    ) + weight * kl_loss_compute(y_2, y_1, reduce=False)

    loss_pick = loss_pick.cpu().detach()

    ind_sorted = torch.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    num_remember = int(rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss = torch.mean(loss_pick[ind_update])

    return loss


def cross_entropy_loss(outputs, soft_targets):
    mask = (soft_targets != -100).sum(1) > 0
    outputs = outputs[mask]
    soft_targets = soft_targets[mask]
    loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
    return loss
