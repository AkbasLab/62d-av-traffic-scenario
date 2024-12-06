import pyarrow.feather as feather
data = feather.read_feather("out/tuning-data/gamma_cross_eb_left_params.feather")
data.to_csv("out/temp/gamma_cross_eb_left_params.csv")
data2 = feather.read_feather("out/tuning-data/gamma_cross_eb_left_scores.feather")
data2.to_csv("out/temp/gamma_cross_eb_left_scores.csv")
