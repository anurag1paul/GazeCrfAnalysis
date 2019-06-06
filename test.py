import os
import pickle
import pycrfsuite

from crfsuite_data import prepare_data
from reporting import looks_classification_report

with open(os.path.join("data/out", "test.pkl"), "rb") as f:
    test = pickle.load(f)

for i, data in enumerate(test):
    tagger = pycrfsuite.Tagger()
    tagger.open('exp_{}'.format(i))

    y_pred = []
    y_true = []
    for features, ylabel in prepare_data(data):
        y_pred.append(tagger.tag(features))
        y_true.append(ylabel)

    print(looks_classification_report(y_true, y_pred))

