import pandas as pd

def main():
    fn = "scenario_config/cross-gama-params.xlsx"
    df = pd.read_excel(fn)
    features = ["feat", "min", "max", "inc", "uom"]
    df = df[features]
    
    msg = ""
    for i in range(len(df.index)):
        s = df.iloc[i]

        for feat in features:
            if feat != "feat":
                msg += " &"
            x = "%s" % s[feat]
            # print(s)
            if x != "nan":
                msg += " %s" % s[feat]

        msg += "\\\\\n"
        # break
    print(msg)
    return

if __name__ == "__main__":
    main()