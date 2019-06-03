import os
import pickle
import pycrfsuite

from crfsuite_data import get_crf_features
from reporting import looks_classification_report

with open(os.path.join("data/out", "test.pkl"), "rb") as f:
    test_dicts = pickle.load(f)

tagger = pycrfsuite.Tagger()
tagger.open('exp1')

y_pred = []
y_true = []

for features, ylabel in test_dicts:
    y_pred.append(tagger.tag(features))
    y_true.append(ylabel)

print(looks_classification_report(y_true, y_pred))

