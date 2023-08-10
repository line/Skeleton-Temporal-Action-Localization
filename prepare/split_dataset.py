import numpy as np
import dutils
import pandas as pd
from collections import Counter
from tqdm import tqdm
import os
from pandas.core.common import flatten
import argparse

MAX_LEN = 1000
N_CLASS = 4

parser = argparse.ArgumentParser(
        description="Spatial Temporal Graph Convolution Network"
    )
parser.add_argument(
    "--data-root",
    default="dataset/babel_v1.0_sequence/",
    help="the root path of the dataset",
    type=str
)
parser.add_argument(
    "--split",
    default=1,
    help="the split of the dataset",
    type=int
)
parser.add_argument(
    "--output-folder",
    default="dataset/processed_data",
    help="the output folder of the generated data",
    type=str
)
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

def main(data_root):
    train_data = dutils.read_pkl(os.path.join(data_root, "babel_v1.0_train_ntu_sk_ntu-style_preprocessed.pkl"))
    test_data = dutils.read_pkl(os.path.join(data_root, "babel_v1.0_val_ntu_sk_ntu-style_preprocessed.pkl"))

    act2idx = dutils.read_json(f"./prepare/configs/action_label_split{args.split}.json")

    label_train_data(data_root, train_data, act2idx)
    label_val_data(data_root, test_data, act2idx)


def label_train_data(data_root, train_data, act2idx):
    sid = []
    x = []
    y = []
    loc = []

    for i, seq_labels in enumerate(tqdm(train_data["Y"])):
        if len(seq_labels) > MAX_LEN:
            continue
        
        y_ = []
        loc_ = []
        flag = False

        for frame, labels in seq_labels.items():
            label_set = set(labels) & set(act2idx.keys())
            label_list = list(label_set)
            if len(label_list) > 0:
                flag = True
                loc_.append(act2idx[label_list[0]])
                y_.append(act2idx[label_list[0]])
            else:
                loc_.append(N_CLASS)

        max_t = len(loc_)
        loc_ = np.array(loc_)
        y_ = list(set(y_))

        if flag:

            # print (train_data["X"][i].shape, len(loc_))
            loc.append(loc_)
            sid.append(train_data["sid"][i])
            x.append(train_data["X"][i][:,:max_t,...])
            y.append(y_)
            
    data = {"sid": sid, "X": x, "Y": y, "L":loc}

    dutils.write_pkl(data, os.path.join(args.output_folder, f"train_split{args.split}.pkl"))
    print (f"#Train sequence: {len(x)}")


def label_val_data(data_root, test_data, act2idx):
    sid = []
    x = []
    y = []
    loc = []
    for i, seq_labels in enumerate(tqdm(test_data["Y"])):
        if len(seq_labels) > MAX_LEN:
            continue

        y_ = []
        loc_ = []
        flag = False

        for frame, labels in seq_labels.items():
            label_set = set(labels) & set(act2idx.keys())
            label_list = list(label_set)
            if len(label_list) > 0:
                flag = True
                loc_.append(act2idx[label_list[0]])
                y_.append(act2idx[label_list[0]])
            else:
                loc_.append(N_CLASS)

        max_t = len(loc_)
        loc_ = np.array(loc_)
        y_ = list(set(y_))

        if flag:
            loc.append(loc_)
            sid.append(test_data["sid"][i])
            x.append(test_data["X"][i][:,:max_t,...])
            y.append(y_)
            
    data = {"sid": sid, "X": x, "Y": y, "L":loc}

    dutils.write_pkl(data, os.path.join(args.output_folder, f"val_split{args.split}.pkl"))
    print (f"#Test sequence: {len(x)}")


if __name__ == "__main__":
    main(args.data_root)
