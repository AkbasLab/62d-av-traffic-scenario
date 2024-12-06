from collision_model import CollisionDataLoader
from lime_ex import LimeAnalyzer
from shap_ex import ShapAnalyzer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define paths for all datasets
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

OUTPUT_DIR = "out/explainability"
Y_COLUMN = "num_collisions"
RANDOM_STATE = None

def evaluate_model_performance(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n=== Model Performance ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    return predictions

def analyze_random_red_light_scenario(X_train, X_test, scenario_test, y_test, model, test_pred):
    print("\n=== Analyzing Random Red Light Running Case ===")

    # Get all red light cases and pick one randomly
    red_light_cases = scenario_test[scenario_test['run_red_light'] == True].index
    if len(red_light_cases) == 0:
        print("No red light cases found in test set")
        return

    random_idx = np.random.choice(red_light_cases)
    print(f"Selected red light scenario: {random_idx}")

    # Use same index for both analyses
    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "red_light")

def analyze_random_side_move_scenario(X_train, X_test, scenario_test, y_test, model, test_pred):
    print("\n=== Analyzing Random Side Move Case ===")

    side_move_cases = scenario_test[scenario_test['side_move'] == True].index
    if len(side_move_cases) == 0:
        print("No side move cases found in test set")
        return

    random_idx = np.random.choice(side_move_cases)
    print(f"Selected side move scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "side_move")

def analyze_random_normal_scenario(X_train, X_test, scenario_test, y_test, model, test_pred):
    print("\n=== Analyzing Random Normal Case ===")

    normal_cases = scenario_test[
        (scenario_test['run_red_light'] == False) &
        (scenario_test['side_move'] == False)
    ].index

    if len(normal_cases) == 0:
        print("No normal cases found in test set")
        return

    random_idx = np.random.choice(normal_cases)
    print(f"Selected normal scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "normal")

def analyze_scenario(X_train, X_test, y_test, test_pred, model, scenario_idx, scenario_type):
    lime_analyzer = LimeAnalyzer(X_train, model, y=Y_COLUMN)
    lime_analyzer.analyze_specific_scenario(
        X_test, y_test, test_pred, scenario_idx,
        output_dir=f"{OUTPUT_DIR}/lime/{scenario_type}",
        verbose=True
    )

    shap_analyzer = ShapAnalyzer(model, X_train)
    shap_analyzer.analyze_specific_scenario(
        X_test, y_test, scenario_idx,
        output_dir=f"{OUTPUT_DIR}/shap/{scenario_type}"
    )

def main():
    print("Loading and combining datasets...")
    full_df = CollisionDataLoader.combine_datasets_with_movement(DATASETS)

    print(f"Dataset shape: {full_df.shape}")
    print("\nFeature names:")
    feature_cols = [col for col in full_df.columns if col not in ['num_collisions', 'run_red_light', 'side_move']]
    print('\n'.join(feature_cols))

    X_train, X_test, y_train, y_test, scenario_train, scenario_test = CollisionDataLoader.split_data(
        full_df,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='regression',
        metric='rmse',
        verbosity=-1
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)
    test_pred = evaluate_model_performance(model, X_test, y_test)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    print("\nTop 10 important features:")
    print(importance.sort_values('importance', ascending=False).head(10))

    analyze_random_red_light_scenario(X_train, X_test, scenario_test, y_test, model, test_pred)
    analyze_random_side_move_scenario(X_train, X_test, scenario_test, y_test, model, test_pred)
    analyze_random_normal_scenario(X_train, X_test, scenario_test, y_test, model, test_pred)

if __name__ == "__main__":
    main()
