import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import utils
import constants

# Show all columns when printing
pd.set_option('display.max_columns', None)

class EDA:
    def __init__(self):
        self.load_data()
        self.count_side_moves()
        self.count_run_red_light()

        for dir in constants.directions:
            for tar in [constants.MONTE_CARLO, constants.SIDE_MOVE]:
                df = self.all_data[dir][tar][constants.SCORES]
                n = df[df.index == len(df.index)-1]["n side move"].iloc[0]
                print(dir, tar, n)

        feature = "n run red light"
        plt.clf()
        fig = plt.figure(figsize=(12,3))
        ax = fig.gca()
        self.feature_comparison(
            feature,
            targets = [constants.RUN_RED_LIGHT, constants.MONTE_CARLO],
            ax = ax
        )
        ax.set_ylabel("# run red light = True")
        plt.savefig(
            "out/graphs/%s_comparison.pdf" % feature, 
            bbox_inches="tight"
        )

        feature = "n side move"
        plt.clf()
        fig = plt.figure(figsize=(12,3))
        ax = fig.gca()
        self.feature_comparison(
            feature,
            targets = [constants.SIDE_MOVE, constants.MONTE_CARLO],
            ax = ax
        )
        ax.set_ylabel("# side move = True")
        plt.savefig(
            "out/graphs/%s_comparison.pdf" % feature, 
            bbox_inches="tight"
        )
        return
    
    def feature_comparison(self, 
            feature : str, 
            targets : list[str], 
            ax : Axes = None
        ):
        """
        Compare Side Move tests with Monte Carlo
        """
        all_data = self.all_data.copy()


        """
        Prepare Graph
        """
        if ax is None:
            plt.clf()
            fig = plt.figure(figsize=(8,3))
            ax = fig.gca()
        
        ax.set_xlabel("# tests")
        ax.set_ylabel("# %s" % feature)
        ax.set_xticks(range(0,10_001,1000))
        ax.set_xlim([0,10_000])
        ax.grid(alpha=.5)

        """
        Count side moves
        """
        max_y = 0
        for target in targets:
            for dir in constants.directions:
                # Collect Side Move data
                df = all_data[dir][target][constants.SCORES]

                # Plot
                x = df.index.tolist()
                y = df[feature]
                max_y = max(max_y, y.max())
                ax.plot(
                    x,y, 
                    linestyle = constants.graphics.linestyles[dir],
                    marker = constants.graphics.markers[target],
                    markevery = 1000,
                    c = "black",
                    label="%s %s" % (constants.graphics.labels[target], dir)
                )
                # break
                continue
            continue
        
        """
        Other Graphics
        """
        ax.set_yticks(range(0,max_y+1,1000))
        ax.set_ylim([0,max_y])
        ax.legend()
        

        
        return ax
    
    def count_run_red_light(self):
        for dir in constants.directions:
            for tar in constants.targets:
                df = self.all_data[dir][tar][constants.SCORES]
                n_run_red_light = []
                n = 0
                for flag in df["run red light"]:
                    if flag:
                        n += 1
                    n_run_red_light.append(n)
                df["n run red light"] = n_run_red_light
        return

    def count_side_moves(self):
        for dir in constants.directions:
            for tar in constants.targets:
                df = self.all_data[dir][tar][constants.SCORES]
                n_side_move = []
                n = 0
                for time in df["side move"]:
                    if time >= 0:
                        n += 1
                    n_side_move.append(n)
                df["n side move"] = n_side_move

                # self.all_data[dir][tar][constants.SCORES]
        return

    def load_data(self):
        self.all_data = {}
        for direction in constants.directions:
            dir_data = {}
            for target in constants.targets:
                params_df = pd.read_feather(
                    "out/%s/%s_gamma_cross_a_eb_%s_params.feather" % (
                        target, target, direction
                    )
                )
                scores_df = pd.read_feather(
                    "out/%s/%s_gamma_cross_a_eb_%s_scores.feather" % (
                        target, target, direction
                    )
                )
                
                dir_data[target] = {
                    "params" : params_df,
                    "scores" : scores_df
                }
                continue
            self.all_data[direction] = dir_data
            continue
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