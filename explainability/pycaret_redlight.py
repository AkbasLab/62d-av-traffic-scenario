from collision_model import CollisionDataLoader
import pandas as pd
import numpy as np
import time
from pycaret import regression

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
Y_COLUMN = "run_red_light"

def log_model_config(model):
   print("\nModel Configuration:")
   print(f"Model params: {model.get_params()}")
   if hasattr(model, 'feature_name_'):
       print(f"Feature names: {model.feature_name_}")

def log_predictions(predictions):
   print("\nPrediction Statistics:")
   print(f"Mean prediction: {predictions['prediction_label'].mean():.4f}")
   print(f"Min prediction: {predictions['prediction_label'].min():.4f}")
   print(f"Max prediction: {predictions['prediction_label'].max():.4f}")

def main():
   print("Loading and combining datasets")
   model_df = CollisionDataLoader.combine_datasets_for_redlight(DATASETS)

   # Convert boolean to int for regression
   model_df[Y_COLUMN] = model_df[Y_COLUMN].astype(int)

   print(f"\nCombined dataset shape: {model_df.shape}")
   print(f"Number of features: {model_df.shape[1] - 1}")  # -1 for target
   print("\nFeature names:")
   feature_cols = [col for col in model_df.columns if col != Y_COLUMN]
   print('\n'.join(feature_cols))

   print("\nSetting up experiment...")
   setup_config = {
       'data': model_df,
       'target': Y_COLUMN,
       'session_id': None,
       'normalize': True,
       'transform_target': False,
       'remove_outliers': False,
       'polynomial_features': False,
       'feature_selection': False,
       'verbose': False,
       'fold': 5,
   }
   print("Setup configuration:")
   print(setup_config)

   regression.setup(**setup_config)

   print("\nPhase 1: Quick Models")
   fast_models = [
       'lr',        # Linear Regression
       'ridge',     # Ridge Regression
       'dt',        # Decision Tree
       'rf'         # Random Forest
   ]
   best_model = regression.compare_models(
       include=fast_models,
       sort='R2',
       n_select=1,
       cross_validation=True,
       fold=5,
       verbose=True
   )
   print(f"\nBest quick model: {best_model}")
   print("\nQuick model details:")
   log_model_config(best_model)

   print("\nEvaluating best quick model:")
   regression.evaluate_model(best_model)

   print("\nPhase 2: Complex Models")
   complex_models = [
       'gbr',       # Gradient Boosting
       'xgboost',   # XGBoost
       'lightgbm',  # LightGBM
       'catboost'   # CatBoost
   ]
   try:
       best_complex = regression.compare_models(
           include=complex_models,
           sort='R2',
           n_select=1,
           cross_validation=True,
           fold=5,
           verbose=True
       )
       print(f"\nBest complex model: {best_complex}")
       print("\nComplex model details:")
       log_model_config(best_complex)

       print("\nEvaluating best complex model:")
       regression.evaluate_model(best_complex)
       # use the best complex model since its already sorted by R^2
       best_model = best_complex
   except Exception as e:
       print(f"Error in complex models phase: {e}")
       print("Continuing with best quick model")

   print("\nMaking predictions")
   predictions = regression.predict_model(best_model, data=model_df)
   log_predictions(predictions)

   print("\nSaving results")
   regression.save_model(best_model, 'best_redlight_model')
   predictions.to_csv('out/explainability/redlight_predictions.csv', index=False)

   print("\nFinal model configuration:")
   log_model_config(best_model)

   print("\nPyCaret internal settings:")
   pycaret_config = regression.get_config('X_train').head()
   print(f"Training data sample:\n{pycaret_config}")

   return best_model, predictions

if __name__ == "__main__":
   start_time = time.time()
   best_model, predictions = main()
   end_time = time.time()
   print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
