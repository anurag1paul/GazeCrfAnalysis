import pandas as pd


rev_lbls = {
            1: "card",
            2: "face",
            3: "dice",
            4: "key",
            5: "map",
            6: "ball",
            0: "none"
            }


def get_crf_features(out):
    df = pd.DataFrame(out)
    features = []
    ylabel = []
    for i in range(1, len(df)):
        frame = df.iloc[i]
        last_frame = df.iloc[i-1]
        f = {
             "bias":1.0,
             "card":frame["card"],
             "dice":frame["dice"],
             "map":frame["map"],
             "key":frame["key"],
             "face":frame["face"],
             "ball":frame["ball"],
             "is_card":frame["card"] != 0,
             "is_dice":frame["dice"] != 0,
             "is_map":frame["map"] != 0,
             "is_key":frame["key"] != 0,
             "is_face":frame["face"] != 0,
             "is_ball":frame["ball"] != 0,
             "prev_card":last_frame["card"],
             "prev_dice":last_frame["dice"],
             "prev_map":last_frame["map"],
             "prev_key":last_frame["key"],
             "prev_face":last_frame["face"],
             "prev_ball":last_frame["ball"]
            }
        features.append(f)
        if frame["look"] != 0:
            ylabel.append(rev_lbls[frame["look"]])
        else:
            ylabel.append("None")
    return features, ylabel

