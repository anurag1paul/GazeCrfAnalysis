import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from constants import SHIFT, SEQ_LEN, REV_LABELS


def get_crf_features(out):
    df = pd.DataFrame(out)
    features = []
    ylabel = []
    y_num_label = []
    for i in range(1, len(df)):
        frame = df.iloc[i]
        last_frame = df.iloc[i-1]
        f = {
             "bias":1.0,
             "card":frame["card"],
             "dice":frame["dice"],
             "map":frame["map"],
             "key":frame["key"],
             "face":frame["face"],
             "ball":frame["ball"],
             "is_card":frame["card_bbox"],
             "is_dice":frame["dice_bbox"],
             "is_map":frame["map_bbox"],
             "is_key":frame["key_bbox"],
             "is_face":frame["face_bbox"],
             "is_ball":frame["ball_bbox"],
             "prev_card":last_frame["card"],
             "prev_dice":last_frame["dice"],
             "prev_map":last_frame["map"],
             "prev_key":last_frame["key"],
             "prev_face":last_frame["face"],
             "prev_ball":last_frame["ball"]
            }
        features.append(f)
        ylabel.append(REV_LABELS[frame["look"]])
        y_num_label.append(frame["look"])
    return features, ylabel, y_num_label


def get_label(arr):
    values, counts = np.unique(arr, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]


def prepare_data(data_dicts):
    features_list = []
    ylabel_list = []
    num_label = []

    for fdict in data_dicts:
        features, ylabel, y_num_label = get_crf_features(fdict)
        features_list.append(features)
        ylabel_list.append(ylabel)
        num_label.append(y_num_label)

    data = []
    for features, ylabel, nlabel in zip(features_list, ylabel_list, num_label):
        for i in range(0, len(features), SHIFT):
            data.append((features[i:i+SEQ_LEN], ylabel[i:i+SEQ_LEN]))

    return data
