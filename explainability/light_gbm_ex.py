from collision_model import CollisionDataLoader
from pycaret.regression import *
import pandas as pd
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
import time

# Define paths for all datasets with their movement types
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

def main():
    # Load and combine datasets
    print("Loading and combining datasets...")
    data_df = CollisionDataLoader.combine_datasets_with_movement(DATASETS)
    print(f"Combined dataset shape: {data_df.shape}")

    # Initialize setup
    print("Setting up experiment...")
    s = setup(
        data=data_df,
        target="num_collisions",
        session_id=42,
        normalize=True,
        transform_target=False,
        remove_outliers=False,
        polynomial_features=False,
        feature_selection=False,
        verbose=False
    )

    # Define parameter grid
    param_grid = {
        'num_leaves': [31, 63],
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_samples': [20, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_split_gain': [0.1]
    }

    # Create base model
    model = LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0,
        importance_type='gain'
    )

    # Create and train model with grid search
    print("\nStarting Grid Search...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        error_score='raise'
    )

    # Get training data
    X = data_df.drop("num_collisions", axis=1)
    y = data_df["num_collisions"]

    # Fit grid search
    grid_search.fit(
        X, y,
        eval_set=[(X, y)],
        eval_metric='rmse',
        callbacks=[None]  # Disable early stopping
    )

    # Print results
    print("\nBest parameters:", grid_search.best_params_)
    print("Best R² score:", grid_search.best_score_)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_model = LGBMRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
    best_model.fit(X, y)

    # Save model and parameters
    print("\nSaving results...")
    pd.DataFrame([grid_search.best_params_]).to_csv('best_parameters.csv', index=False)
    save_model(best_model, 'best_lightgbm_model')

    return best_model, grid_search.best_params_

if __name__ == "__main__":
    start_time = time.time()
    best_model, best_params = main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
