import pandas as pd
import numpy as np
import os
import ast
import math
import pickle

from collections import defaultdict


root_dir = "data/"
object_det = os.path.join(root_dir, "object_detections")
gaze = os.path.join(root_dir, "gaze")
gt = os.path.join(root_dir, "annotations")


def read_object_detections(filename):

    with open(os.path.join(object_det, filename)) as f:
        object_data = f.readlines()

    frame_list = []
    for line in object_data:
        frame = defaultdict(list)

        if line != "\n":
            a = ast.literal_eval(line)
            if isinstance(a, tuple):
                for b in a:
                    try:
                        obj = ast.literal_eval(b)
                        frame[obj[0]].append((obj[2][0]/1280, 1 - obj[2][1]/720, obj[1]))
                    except:
                        print(a)
            else:
                obj = ast.literal_eval(a)
                frame[obj[0]].append((obj[2][0]/1280, 1 - obj[2][1]/720, obj[1]))

        frame_list.append(frame)
    return frame_list


def read_gaze_file(filename):
    gaze_data = pd.read_csv(os.path.join(gaze, filename))
    gaze_data = gaze_data[["timestamp", "index", "confidence", "norm_pos_x", "norm_pos_y"]]
    max_idx = int(gaze_data.iloc[-1]["index"] + 1)
    tpf = (gaze_data.iloc[-1]["timestamp"] - gaze_data.iloc[0]["timestamp"]) / max_idx
    gaze_list = []
    for i in range(max_idx):
        frame = gaze_data[gaze_data["index"] == i]
        filtered = frame[frame["confidence"] > 0.5]
        x = filtered["norm_pos_x"].mean()
        y = filtered["norm_pos_y"].mean()
        if math.isnan(x) or math.isnan(y):
            x, y = gaze_list[-1]
        gaze_list.append((x, y))

    return gaze_list, max_idx, tpf


def get_frame_gaze_dict(gaze_list, frame_list, max_idx):
    out = {"card":[],
          "face":[],
          "dice":[],
          "key":[],
          "map":[],
          "ball":[]}

    for i in range(max_idx):
        pupil = gaze_list[i]
        frame = frame_list[i]
        for key in out.keys():
            if key in frame:
                dists = []
                for pt in frame[key]:
                    dist = ((pupil[0] - pt[0])**2 + (pupil[1] - pt[1])**2)**0.5
                    if math.isnan(dist):
                        print(pupil, pt)
                    else:
                        dists.append(dist)
                min_dist = min(dists)
                out[key].append((2-min_dist)/2)
            else:
                out[key].append(0)

    out["index"] = np.arange(max_idx)
    return out


def read_looks_gt_file(filename, out, max_idx, tpf):

    looks = pd.read_csv(os.path.join(gt, filename))[["object", "start_sec", "end_sec"]]
    looks = looks.sort_values("start_sec")
    out["look"] = np.array([0]*max_idx)
    labels = {"card":1,
          "face":2,
          "dice":3,
          "key":4,
          "map":5,
          "ball":6}
    for row in looks.values:
        start = int(np.floor(row[1] / tpf))
        end = int(np.ceil(row[2] / tpf))
        out["look"][start:end] = labels[row[0]]

    out["look"] = list(out["look"])
    return out


def main():

    obj_det_files = ["Andy.csv", "Daniel.csv", "2018_07_17_001.csv", "2018_07_24_003.csv", "2018_07_17_004.csv"]
    gaze_files = ["Andy_gaze_positions.csv", "Daniel_gaze_positions.csv",
                  "2018_07_17_001_gaze_positions.csv", "2018_07_24_003_gaze_positions.csv",
                  "2018-07-17-004_gaze_positions.csv"]
    gt_files = ["Andy_annotated.csv", "daniel_annotated.csv", "7_17_001_annotated.csv",
                "7_24_003_annotated.csv", "2018-07-17-004_annotated.csv"]

    obj_data = []

    for file in obj_det_files:
        obj_data.append(read_object_detections(file))

    gaze_data = []
    max_idxs = []
    tpfs = []

    for file in gaze_files:
        gz, m_idx, tpf = read_gaze_file(file)
        gaze_data.append(gz)
        max_idxs.append(m_idx)
        tpfs.append(tpf)

    out_dicts = []
    for obj, gz, midx in zip(obj_data, gaze_data, max_idxs):
        out_dicts.append(get_frame_gaze_dict(gz, obj, midx))

    final_dicts = []
    for file, out, midx, tpf in zip(gt_files, out_dicts, max_idxs, tpfs):
        final_dicts.append(read_looks_gt_file(file, out, midx, tpf))

    out_dir = os.path.join("data", "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "data.pkl"), "wb") as f:
        pickle.dump(final_dicts, f)

if __name__ == "__main__":
    main()

