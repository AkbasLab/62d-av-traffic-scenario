from collision_model import CollisionDataLoader
from lime_ex import LimeAnalyzer
from shap_ex import ShapAnalyzer
import numpy as np
import pandas as pd
import os
from pycaret import classification

DATASETS = [
   ("out/full_data/full_data_gamma_cross_a_eb_left_params.feather",
    "out/full_data/full_data_gamma_cross_a_eb_left_scores.feather",
    "left"),
   ("out/full_data/full_data_gamma_cross_a_eb_straight_params.feather",
    "out/full_data/full_data_gamma_cross_a_eb_straight_scores.feather",
    "straight"),
   ("out/full_data/full_data_gamma_cross_a_eb_right_params.feather",
    "out/full_data/full_data_gamma_cross_a_eb_right_scores.feather",
    "right")
]

OUTPUT_DIR = "out/explainability/side_move"
Y_COLUMN = "side_move"
RANDOM_STATE = None

def analyze_scenario(X_train, X_test, y_test, test_pred, model, scenario_idx, position):
    """Analyze a specific scenario with both Lime and SHAP"""
    lime_dir = os.path.join(OUTPUT_DIR, "local", "lime")
    shap_dir = os.path.join(OUTPUT_DIR, "local", "shap")

    actual_value = y_test[scenario_idx]
    predicted_value = test_pred[position]

    lime_analyzer = LimeAnalyzer(X_train, model, y=Y_COLUMN)
    lime_analyzer.analyze_specific_scenario(
        X_test, y_test, test_pred, scenario_idx,
        output_dir=lime_dir,
        verbose=True
    )

    shap_analyzer = ShapAnalyzer(model, X_train)
    shap_analyzer.analyze_specific_scenario_sidemove(
        X_test, y_test, scenario_idx,
        output_dir=shap_dir,
        actual_value=actual_value,
        predicted_value=predicted_value
    )

def main():
   print("Loading and combining datasets")
   model_df = CollisionDataLoader.combine_datasets_for_sidemove(DATASETS)

   print(f"Dataset shape: {model_df.shape}")
   print("\nFeature names:")
   feature_cols = [col for col in model_df.columns if col != Y_COLUMN]
   print('\n'.join(feature_cols))

   print("\nSetting up PyCaret environment")
   classification.setup(
       data=model_df,
       target=Y_COLUMN,
       session_id=RANDOM_STATE,
       normalize=True,
       remove_outliers=False,
       polynomial_features=False,
       feature_selection=False,
       fold=5,
       verbose=False
   )

   print("\nTraining LightGBM model")
   model = classification.create_model('lightgbm')

   if not hasattr(model, 'predict'):
       raise ValueError("Extracted model doesn't have specified prediction method")

   X_train = classification.get_config('X_train')
   X_test = classification.get_config('X_test')
   y_test = classification.get_config('y_test')

   print("\n___ Model Evaluation ___")
   test_pred = model.predict(X_test)

   print("\nCalculating feature importance")
   if hasattr(model, 'feature_name_'):
       feature_names = model.feature_name_
   else:
       feature_names = feature_cols

   importance = pd.DataFrame({
       'feature': feature_names,
       'importance': model.feature_importances_
   })

   output_path = os.path.join(OUTPUT_DIR, "global", "lightgbm")
   os.makedirs(output_path, exist_ok=True)
   importance.to_csv(
       os.path.join(output_path, "lightgbm_global_feature_ranking.txt"),
       index=False,
       sep='\t',
       float_format='%.4f'
   )

   importance_sorted = importance.sort_values('importance', ascending=False)
   print("\nTop feature importance ranking:")
   print(importance_sorted)

   print("\nRunning global SHAP analysis...")
   shap_analyzer = ShapAnalyzer(model, X_train)
   shap_analyzer.analyze_global_importance(
       X_train,
       output_dir=os.path.join(OUTPUT_DIR, "global", "shap")
   )

   print("\nAnalyzing random case where side move occurred...")
   # Find indices where side move occurred
   side_move_positions = np.where(y_test)[0]
   if len(side_move_positions) == 0:
       print("No side move cases found in test set")
       return

   # Get random side move case
   position = np.random.choice(side_move_positions)
   random_idx = X_test.index[position]
   actual_value = y_test[random_idx]
   predicted_value = test_pred[position]
   print(f"Selected scenario: {random_idx}")
   print(f"Side move occurred: {actual_value}")
   print(f"Predicted side move: {predicted_value}")

   analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, position)

if __name__ == "__main__":
   main()
