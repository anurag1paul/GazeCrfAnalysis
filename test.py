import os
import pickle
import pycrfsuite

from crfsuite_data import prepare_data
from reporting import StatsManager, pretty_print_report, pretty_rl_table

with open(os.path.join("data/out", "test.pkl"), "rb") as f:
    test = pickle.load(f)

support_threshold = 100
stats = StatsManager(support_threshold)

for i, data in enumerate(test):
    tagger = pycrfsuite.Tagger()
    tagger.open('exp_{}'.format(i))

    y_pred = []
    y_true = []
    for features, ylabel in prepare_data(data):
        y_pred.append(tagger.tag(features))
        y_true.append(ylabel)

    stats.append_report(y_true, y_pred)

print("Multi-class Classification Report Mean(Std)")
report, summary = stats.summarize()
pretty_print_report(report)

print("Run-length Report")
rl_report = stats.runlength_report()
pretty_rl_table(rl_report)
