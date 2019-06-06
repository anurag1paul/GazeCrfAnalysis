import os

# constants to be used for the project
ROOT = "data/"
OBJECT_DET_DIR = os.path.join(ROOT, "object_detections")
GAZE_POS_DIR = os.path.join(ROOT, "gaze")
GT_DIR = os.path.join(ROOT, "annotations")



OBJECT_DETECTIONS = ["Andy.csv", "Daniel.csv", "2018_07_17_001.csv",
                     "2018_07_24_003.csv", "2018_07_17_004.csv"]  # "2018_07_25_000.csv"]
GAZE_POSITIONS = ["Andy_gaze_positions.csv", "Daniel_gaze_positions.csv",
                  "2018_07_17_001_gaze_positions.csv", "2018_07_24_003_gaze_positions.csv",
                  "2018-07-17-004_gaze_positions.csv"]  # "2018_07_25_gaze_positions.csv"]
GROUND_TRUTH = ["Andy_annotated.csv", "daniel_annotated.csv", "7_17_001_annotated.csv",
                "7_24_003_annotated.csv", "2018-07-17-004_annotated.csv"]  # "7_25_000_annotated.csv"]

FILE_CHUNK = 900

# CV related
NUMBER_OF_CV_FOLDS = 5

# Feature Related
SEQ_LEN = 10
SHIFT = 5

REV_LABELS = {
            1: "card",
            2: "face",
            3: "dice",
            4: "key",
            5: "map",
            6: "ball",
            0: "none"
            }

LABELS = {"card": 1,
          "face": 2,
          "dice": 3,
          "key": 4,
          "map": 5,
          "ball": 6,
          "none": 0}
