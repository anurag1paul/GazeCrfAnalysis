import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

rev_lbls = {
            1: "card",
            2: "face",
            3: "dice",
            4: "key",
            5: "map",
            6: "ball",
            0: "none"
            }


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
        ylabel.append(rev_lbls[frame["look"]])
        y_num_label.append(frame["look"])
    return features, ylabel, y_num_label

def get_label(arr):
    values, counts = np.unique(arr, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]

def main():
    with open(os.path.join("data/out", "data.pkl"), "rb") as f:
        final_dicts = pickle.load(f)

    features_list = []
    ylabel_list = []
    num_label = []

    for fdict in final_dicts:
        features, ylabel, y_num_label = get_crf_features(fdict)
        features_list.append(features)
        ylabel_list.append(ylabel)
        num_label.append(y_num_label)

    data = []
    labels = []
    for features, ylabel, nlabel in zip(features_list, ylabel_list, num_label):
        for i in range(0, len(features), 5):
            data.append((features[i:i+20], ylabel[i:i+20]))
            labels.append(get_label(nlabel[i:i+20]))
    
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    
    with open(os.path.join("data/out", "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    
    with open(os.path.join("data/out", "test.pkl"), "wb") as f:
        pickle.dump(test, f)

if __name__ == "__main__":
    main()
    print("Train and Test Subsets built")
