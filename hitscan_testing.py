import os
import pickle
import pycrfsuite

from crfsuite_data import get_crf_features
from reporting import looks_classification_report

with open(os.path.join("data/out", "test.pkl"), "rb") as f:
    test_dicts = pickle.load(f)

def ftag(features):
    pred = []
    for f in features:
        p = "none"
        if f['is_card']:
            p = "card"
        elif f['is_dice']:
            p="dice"
        elif f['is_map']:
            p="map"
        elif f['is_ball']:
            p="ball"
        elif f['is_face']:
            p="face"
        elif f['is_key']:
            p="key"
        pred.append(p)
    return pred
    
    
y_pred = []
y_true = []

for features, ylabel in test_dicts:
    y_pred.append(ftag(features))
    y_true.append(ylabel)

print(looks_classification_report(y_true, y_pred))

