import pycrfsuite
import os
import pickle

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

trainer = pycrfsuite.Trainer(verbose=False)

for features, ylabel in zip(features_list[:-1], ylabel_list[:-1]):
    for i in range(0, len(features), 5):
        trainer.append(features[i:i+20], ylabel[i:i+20])

trainer.set_params({'c1': 0.1,   # coefficient for L1 penalty
                    'c2': 0.1,  # coefficient for L2 penalty
                    'max_iterations': 100000,  # stop earlier
                    'feature.possible_transitions': True})

trainer.train("exp1")
