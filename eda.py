import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils
import constants

# Show all columns when printing
pd.set_option('display.max_columns', None)

class EDA:
    def __init__(self):
        direction = "right"
        self.params_df = pd.read_feather(
                "out/mc/mc_gamma_cross_a_eb_%s_params.feather" % direction)
        self.scores_df = pd.read_feather(
            "out/mc/mc_gamma_cross_a_eb_%s_scores.feather" % direction)
        
        self.mc_stats()
        return
    
    def mc_stats(self):
        df = self.scores_df.copy()
        df = df.round(decimals=5)

        df["n collisions"] = df["collisions"].apply(len)
        df["n foes in inter (on enter)"] = df["foes in inter (on enter)"]\
            .apply(len)

        df.drop(
            columns=["collisions", "foes in inter (on enter)"], 
            inplace=True
        )

        for feat in ["speed (on enter)", "time (on enter)", "side move"]:
            df[feat] = df[feat]\
                .apply(lambda x: x if x >= 0 else None)
            
        df["speed (on enter)"].apply(utils.mps2kph)

        # df["side move"] = df["side move"]\
        #     .apply(lambda x : 1 if x >= 0 else None)
        
        df["run red light"] = df["run red light"].astype(int)
        df["tl state (on enter)"] = df["tl state (on enter)"].apply(
            lambda x : constants.traci.gamma_cross.tl_state[x]
        )
        
        for feat in ["dtc (front)", "ttc (front)", "dtc (inter)", 
                     "dtc (approach)"]:
            df[feat] = df[feat]\
                .apply(lambda x: x if x < 9999 else None)
            
        utils.describe_as_latex(df)
        # df = df.describe().T[["count", "mean", "std", "min", "max"]]
        
        return
    
    def head_100(self):
        scores_df = self.scores_df
        print(scores_df.iloc[:10])
        print()

        # scores_df.iloc[:100].to_csv("temp/scores.tsv", sep="\t")

        # collision = scores_df.iloc[3]["collisions"][0]
        # print(collision)
        return

    def quick_look(self):
        for direction in ["left", "straight"]:
            print("\n\n")
            print(":: %s TURN ::" % direction.upper())
            print()

            params_df = pd.read_feather(
                "out/mc/mc_gamma_cross_eb_%s_params.feather" % direction)
            scores_df = pd.read_feather(
                "out/mc/mc_gamma_cross_eb_%s_scores.feather" % direction)
            
            
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