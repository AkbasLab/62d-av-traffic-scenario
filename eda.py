import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt


class EDA:
    def __init__(self):
        for direction in ["left", "straight", "right"]:
            print("\n\n")
            print(":: %s TURN ::" % direction.upper())
            print()

            params_df = pd.read_feather(
                "out/tuning-data/gamma_cross_eb_%s_params.feather" % direction)
            scores_df = pd.read_feather(
                "out/tuning-data/gamma_cross_eb_%s_scores.feather" % direction)
            
            
            df = scores_df

            # Quantify Scores
            df["n collisions"] = df["collisions"].apply(len)
            df["n foes in inter (on enter)"] = \
                df["foes in inter (on enter)"].apply(len)
            df["run red light"] = df["run red light"].apply(int)

            # Filter NA
            for feat in df.columns.to_list():
                try:
                    df[feat] = df[feat].apply(
                        lambda x: None if (x < 0 or x > 9000) else x)
                except TypeError:
                    pass
            # print(df.iloc[0])
            df = scores_df.describe().T[["count","mean","std","min","max"]]
            print(df)
        return
    
if __name__ == "__main__":
    EDA()