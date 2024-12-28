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
        # self.load_data()
        # self.compare_targeted_testing()
        # self.stat_summary()

        self.lightbm_feature_importance()
        return
    
    def lightbm_feature_importance(self):
        for tsc in ["num_collisions", "run_red_light", "side_move"]:
            fn = "out/explainability/%s/global/lightgbm/lightgbm_global_feature_ranking.txt" % tsc
            pdf_fn = "%s.pdf" % fn.split(".")[0]
            self.load_and_plot_lightgbm_features(fn)        
            plt.savefig(pdf_fn, bbox_inches = "tight")
        return

    def load_and_plot_lightgbm_features(self, fn : str):
        plt.clf()

        df = pd.read_csv(fn, sep="\t")\
            .sort_values("importance",ascending=False)\
            .reset_index(drop=True)
        
        df["norm"] = df["importance"]/df["importance"].sum()

        df = df.iloc[:10]
        
        figure = plt.figure(figsize=(5,4))
        ax = figure.gca()

        x = df["feature"]
        width = df["norm"]

        ax.barh(
            x, width,
            color="black",
            zorder=2
        )
        ax.invert_yaxis()
        ax.grid(True,zorder=-1)

        ax.set_xlabel("normalized feature impact on model output")
        return ax

    def stat_summary(self):
        # Collect Data
        data = []
        for dir in constants.directions:
            for tar in constants.targets:
                df = self.all_data[dir][tar][constants.SCORES].copy()
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
                
                df["run red light"] = df["run red light"].astype(int)
                df["tl state (on enter)"] = df["tl state (on enter)"].apply(
                    lambda x : constants.traci.gamma_cross.tl_state[x]
                )
                
                for feat in ["dtc (front)", "ttc (front)", "dtc (inter)", 
                            "dtc (approach)"]:
                    df[feat] = df[feat]\
                        .apply(lambda x: x if x < 9999 else None)
                    continue
                # df["tar"] = tar
                df["dir"] = dir
                data.append(df.copy())
                continue
            continue
        df = pd.concat(data)
        df = df.drop(columns=["envelope_id", "stage", "is_target"])

        # Get summary
        data = []
        for dir in constants.directions:
            dir_df = df[df["dir"] == dir].describe().T
            dir_df["dir"] = dir
            dir_df["feature"] = dir_df.index
            dir_df.reset_index(drop=True,inplace=True)
            data.append( dir_df.copy() )    
        df = pd.concat(data)

        print(df)

        # Build the table
        msg = "\\begin{table}[!ht]\n"
        msg += "\t\\centering\n"
        msg += "\t\\caption{}\n"
        msg += "\t\\label{}\n"
        msg += "\t\\begin{tabular}{rllrr@{.}lr@{.}lr@{.}lr@{.}l}\\toprule\n"
        msg += "\t\\multicolumn{2}{l}{Feature} & Scenario & Count & "
        msg += "\\multicolumn{2}{c}{Mean} & \\multicolumn{2}{c}{Std} & "
        msg += "\\multicolumn{2}{c}{Min} & \\multicolumn{2}{c}{Max} \\\\"
        msg += "\\midrule\n"
        
        # Table Data
        for index in range(15):
            idf = df[df.index == index].round(decimals=3)
            left = idf[idf["dir"] == "left"].iloc[0]
            straight = idf[idf["dir"] == "straight"].iloc[0]
            right = idf[idf["dir"] == "right"].iloc[0]

            m = '$M_{%d}$' % index
            if index in [0,10]:
                m = "len(%s)" % m

            # feature = "%s & %s" % (m, left["feature"])

            left_stats = self.stats2latex(left)
            straight_stats = self.stats2latex(straight)
            right_stats = self.stats2latex(right)

            msg += "%s & %s & Left Turn %s\\\\\n" % \
                (m, left["feature"], left_stats)
            msg += "&& Straight Ahead %s\\\\\n" % straight_stats
            msg += "&& Right Turn %s\\\\\n" % right_stats
            # print(msg)
            # quit()
            continue

        # Table footer
        msg += "\\bottomrule\n"
        msg += "\\end{tabular}\n"
        msg += "\\end{table}"

        
        # print(df)
        # print(msg)
        return
    
    def stats2latex(self, s : pd.Series) -> str:
        stats = ["count", "mean", "std", "min", "max"]
        msg = ""
        for feat in stats:
            left , right = str(s[feat]).split(".")
            left = "{:,}".format(int(left))
            if feat == "count":
                msg += "& %s " % left
            else: 
                msg += "& %s&%s " % (left,right)
            continue
        return msg
    
    def compare_targeted_testing(self):
        self.count_side_moves()
        self.count_run_red_light()
        for feat in ["n side move", "n run red light"]:
            print(":: %s ::" % feat)
            data = []
            for dir in constants.directions:
                for tar in constants.targets:
                    df = self.all_data[dir][tar][constants.SCORES]
                    n = df[feat].max()
                    s = pd.Series({
                        "dir" : dir,
                        "tar" : tar,
                        "n" : n
                    })
                    data.append(s)
                    continue
                continue
            feat_df = pd.DataFrame(data)
            print(feat_df)
            continue

            
    
        return
    
    def comparison_graphs(self):
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