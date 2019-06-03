import pycrfsuite
import os
import pickle

from reporting import looks_classification_report

with open(os.path.join("data/out", "train.pkl"), "rb") as f:
    train_dicts = pickle.load(f)

trainer = pycrfsuite.Trainer(verbose=False)

for features, ylabel in train_dicts:
    trainer.append(features, ylabel)

trainer.set_params({'c1': 0.1,   # coefficient for L1 penalty
                    'c2': 0.1,  # coefficient for L2 penalty
                    'max_iterations': 100000,  # stop earlier
                    'feature.possible_transitions': True})

trainer.train("exp1")
print("Model Trained")
