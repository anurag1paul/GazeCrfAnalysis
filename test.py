import os
import pickle
import pycrfsuite

from crfsuite_data import get_crf_features
from reporting import looks_classification_report

with open(os.path.join("data/out", "data.pkl"), "rb") as f:
    final_dicts = pickle.load(f)

features_list = []
ylabel_list = []

for fdict in final_dicts:
    features, ylabel = get_crf_features(fdict)
    features_list.append(features)
    ylabel_list.append(ylabel)

tagger = pycrfsuite.Tagger()
tagger.open('exp1')

y_pred = []
y_true = []

for features, ylabel in zip(features_list[:-1], ylabel_list[:-1]):
    for i in range(0, len(features), 5):
        y_pred.append(tagger.tag(features[i:i+20]))
        y_true.append(ylabel[i:i+20])

print(looks_classification_report(y_true, y_pred))

