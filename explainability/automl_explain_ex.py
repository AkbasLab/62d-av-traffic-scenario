from collision_model import CollisionDataLoader
from pycaret.regression import *
import pandas as pd
import numpy as np
import time

# Define paths for all datasets
DATASETS = [
    ("out/mc/mc_gamma_cross_eb_left_params.feather", "out/mc/mc_gamma_cross_eb_left_scores.feather"),
    ("out/mc/mc_gamma_cross_eb_straight_params.feather", "out/mc/mc_gamma_cross_eb_straight_scores.feather"),
    ("out/mc/mc_gamma_cross_eb_right_params.feather", "out/mc/mc_gamma_cross_eb_right_scores.feather")
]

Y_COLUMN = "num_collisions"

def main():
    # Load and combine all datasets
    print("Loading and combining datasets...")
    data_df = CollisionDataLoader.combine_datasets(
        dataset_paths=DATASETS,
        column_to_use="collisions",
        new_column=Y_COLUMN
    )

    # Print dataset info
    print(f"\nCombined dataset shape: {data_df.shape}")
    print(f"Number of features: {data_df.shape[1] - 1}")  # -1 for target

    # Initialize setup with minimal preprocessing
    print("\nSetting up experiment...")
    s = setup(
        data=data_df,
        target=Y_COLUMN,
        session_id=42,
        normalize=True,
        transform_target=False,
        remove_outliers=False,
        polynomial_features=False,
        feature_selection=False,
        verbose=False,
        fold=5,  # 5-fold CV
    )

    # Rest of your code remains the same...
    print("\nPhase 1: Quick Models")
    fast_models = [
        'lr',        # Linear Regression
        'ridge',     # Ridge Regression
        'dt',        # Decision Tree
        'rf'         # Random Forest
    ]

    best_model = compare_models(
        include=fast_models,
        sort='R2',
        n_select=1,
        cross_validation=True,
        fold=5,
        verbose=True
    )

    print(f"\nBest quick model: {best_model}")
    print("\nEvaluating best quick model:")
    evaluate_model(best_model)

    print("\nPhase 2: Complex Models")
    complex_models = [
        'gbr',       # Gradient Boosting
        'xgboost',   # XGBoost
        'lightgbm',  # LightGBM
        'catboost'   # CatBoost
    ]

    try:
        best_complex = compare_models(
            include=complex_models,
            sort='R2',
            n_select=1,
            cross_validation=True,
            fold=5,
            verbose=True
        )

        print(f"\nBest complex model: {best_complex}")
        print("\nEvaluating best complex model:")
        evaluate_model(best_complex)

        # Choose overall best model
        if get_model_metrics(best_complex)['R2'] > get_model_metrics(best_model)['R2']:
            best_model = best_complex

    except Exception as e:
        print(f"Error in complex models phase: {e}")
        print("Continuing with best quick model...")

    print("\nMaking predictions...")
    predictions = predict_model(best_model, data=data_df)

    print("\nSaving results...")
    save_model(best_model, 'best_collision_model')
    predictions.to_csv('collision_predictions.csv', index=False)

    return best_model, predictions

if __name__ == "__main__":
    start_time = time.time()
    best_model, predictions = main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
