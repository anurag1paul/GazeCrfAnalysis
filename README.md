# GazeCrfAnalysis

- data_processing.py: converts raw_data into formatted dicts and then splits
the data into multiple sets of train and test based on 5-fold cross validation

- crfsuite_data.py: helper functions for formatting data for for Python-crfsuite

- model.py: trains and create model files for each train-test split

- test.py: evaluates the model

- hitscan_testing.py: to compare with the baseline

- reporting.py: helper functions for statistics
