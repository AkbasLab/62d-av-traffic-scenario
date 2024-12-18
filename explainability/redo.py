from collision_model import CollisionDataLoader
import pandas as pd
import numpy as np
import time
from pycaret import regression

DATASETS = [
    ("out/mc/mc_gamma_cross_eb_left_params.feather",
     "out/mc/mc_gamma_cross_eb_left_scores.feather",
     "left"),
    ("out/mc/mc_gamma_cross_eb_straight_params.feather",
     "out/mc/mc_gamma_cross_eb_straight_scores.feather",
     "straight"),
    ("out/mc/mc_gamma_cross_eb_right_params.feather",
     "out/mc/mc_gamma_cross_eb_right_scores.feather",
     "right")
]
Y_COLUMN = "num_collisions"

def main():
    print("Loading and combining datasets...")
    model_df, scenario_df = CollisionDataLoader.combine_datasets(
        dataset_paths=DATASETS,
        add_movement_vars=True
    )

    print(f"\nCombined dataset shape: {model_df.shape}")
    print(f"Number of features: {model_df.shape[1] - 1}")  # -1 for target

    print("\nSetting up experiment...")
    regression.setup(
        data=model_df,
        target=Y_COLUMN,
        session_id=42,
        normalize=True,
        transform_target=False,
        remove_outliers=False,
        polynomial_features=False,
        feature_selection=False,
        fold=5,
        verbose=False
    )

    print("\nTraining LightGBM model...")
    model = regression.create_model('lightgbm')

    print("\nEvaluating model...")
    regression.evaluate_model(model)

    print("\nMaking predictions...")
    predictions = regression.predict_model(model, data=model_df)

    print("\nModel Performance:")
    rmse = np.sqrt(((predictions['prediction_label'] - predictions[Y_COLUMN])**2).mean())
    mae = abs(predictions['prediction_label'] - predictions[Y_COLUMN]).mean()
    r2 = 1 - (((predictions['prediction_label'] - predictions[Y_COLUMN])**2).sum() /
              ((predictions[Y_COLUMN] - predictions[Y_COLUMN].mean())**2).sum())

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return model, predictions

if __name__ == "__main__":
    start_time = time.time()
    model, predictions = main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
