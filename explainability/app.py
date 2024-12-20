from collision_model import CollisionDataLoader
from lime_ex import LimeAnalyzer
from shap_ex import ShapAnalyzer
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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

OUTPUT_DIR = "out/explainability"
Y_COLUMN = "num_collisions"
RANDOM_STATE = None

def analyze_random_red_light_scenario(X_train, X_test, scenario_test, y_test, model, test_pred, output_dir):
    print("\n___ Analyzing Random Red Light Running Case ___")

    red_light_cases = scenario_test[scenario_test['run_red_light'] == True].index
    if len(red_light_cases) == 0:
        print("No red light cases found in test set")
        return

    random_idx = np.random.choice(red_light_cases)
    print(f"Selected red light scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "red_light", output_dir)

def analyze_random_side_move_scenario(X_train, X_test, scenario_test, y_test, model, test_pred, output_dir):
    print("\n___ Analyzing Random Side Move Case ___")

    side_move_cases = scenario_test[scenario_test['side_move'] == True].index
    if len(side_move_cases) == 0:
        print("No side move cases found in test set")
        return

    random_idx = np.random.choice(side_move_cases)
    print(f"Selected side move scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "side_move", output_dir)

def analyze_random_normal_scenario(X_train, X_test, scenario_test, y_test, model, test_pred, output_dir):
    print("\n___ Analyzing Random Normal Case ___")

    normal_cases = scenario_test[
        (scenario_test['run_red_light'] == False) &
        (scenario_test['side_move'] == False)
    ].index

    if len(normal_cases) == 0:
        print("No normal cases found in test set")
        return

    random_idx = np.random.choice(normal_cases)
    print(f"Selected normal scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "normal", output_dir)

def analyze_scenario(X_train, X_test, y_test, test_pred, model, scenario_idx, scenario_type, output_dir):
    lime_dir = os.path.join(output_dir, "local", "lime", scenario_type)
    shap_dir = os.path.join(output_dir, "local", "shap", scenario_type)

    lime_analyzer = LimeAnalyzer(X_train, model, y=Y_COLUMN)
    lime_analyzer.analyze_specific_scenario(
        X_test, y_test, test_pred, scenario_idx,
        output_dir=lime_dir,
        verbose=True
    )

    shap_analyzer = ShapAnalyzer(model, X_train)
    shap_analyzer.analyze_specific_scenario(
        X_test, y_test, scenario_idx,
        output_dir=shap_dir
    )

def main():
    print("Loading and combining datasets")
    model_df, scenario_df = CollisionDataLoader.combine_datasets(DATASETS, add_movement_vars=True)

    print(f"Dataset shape: {model_df.shape}")
    print("\nFeature names:")
    feature_cols = [col for col in model_df.columns if col != 'num_collisions']
    print('\n'.join(feature_cols))

    # First run with all features
    all_features_dir = os.path.join(OUTPUT_DIR, "all_features")
    print("\n___ Running analysis with all features ___")

    print("\nSetting up PyCaret environment")
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

    print("\nTraining LightGBM model with all features")
    full_model = regression.create_model('lightgbm')

    if not hasattr(full_model, 'predict'):
        raise ValueError("Extracted model doesn't have specified prediction method")

    # split data (but keep scenarios aligned)
    train_indices, test_indices = train_test_split(
        model_df.index,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # split model features
    X_train = model_df.loc[train_indices, feature_cols]
    X_test = model_df.loc[test_indices, feature_cols]
    y_test = model_df.loc[test_indices, 'num_collisions']
    scenario_test = scenario_df.loc[test_indices]

    print("\n___ Full Model Evaluation ___")
    test_pred = full_model.predict(regression.get_config('X_test'))


    print("\nCalculating feature importance")
    if hasattr(full_model, 'feature_name_'):
        feature_names = full_model.feature_name_
    else:
        feature_names = feature_cols

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': full_model.feature_importances_
    })

    # save
    output_path = os.path.join(all_features_dir, "global", "lightgbm")
    os.makedirs(output_path, exist_ok=True)
    importance.to_csv(
        os.path.join(output_path, "lightgbm_global_feature_ranking.txt"),
        index=False,
        sep='\t',
        float_format='%.4f'
    )

    # Sort and get top 30
    importance_sorted = importance.sort_values('importance', ascending=False)
    print("\nTop 30 features:")
    print(importance_sorted.head(30))
    top_30_features = importance_sorted.head(30)['feature'].tolist()

    # full model
    print("\nRunning full model SHAP analysis...")
    shap_analyzer = ShapAnalyzer(full_model, X_train)
    shap_analyzer.analyze_global_importance(
        X_train,
        output_dir=os.path.join(all_features_dir, "global", "shap")
    )

    print("\nAnalyzing full model scenarios...")
    analyze_random_red_light_scenario(
        X_train, X_test, scenario_test, y_test, full_model, test_pred, all_features_dir
    )
    analyze_random_side_move_scenario(
        X_train, X_test, scenario_test, y_test, full_model, test_pred, all_features_dir
    )
    analyze_random_normal_scenario(
        X_train, X_test, scenario_test, y_test, full_model, test_pred, all_features_dir
    )

    # train w/ top 30
    top_30_dir = os.path.join(OUTPUT_DIR, "top_30")
    print("\n=== Running analysis with top 30 features ===")

    print("\nSetting up PyCaret environment for reduced feature set...")
    regression.setup(
        data=model_df.loc[:, top_30_features + ['num_collisions']],
        target=Y_COLUMN,
        session_id=43,
        normalize=True,
        transform_target=False,
        remove_outliers=False,
        polynomial_features=False,
        feature_selection=False,
        fold=5,
        verbose=False
    )

    print("\nTraining LightGBM model with top 30 features")
    reduced_model = regression.create_model('lightgbm')

    # reduce features
    X_train_reduced = X_train[top_30_features]
    X_test_reduced = X_test[top_30_features]

    print("\n___ Reduced Model Evaluation ___")
    test_pred_reduced = reduced_model.predict(regression.get_config('X_test'))

    # save reduced
    importance_reduced = pd.DataFrame({
        'feature': top_30_features,
        'importance': reduced_model.feature_importances_
    })

    output_path = os.path.join(top_30_dir, "global", "lightgbm")
    os.makedirs(output_path, exist_ok=True)
    importance_reduced.to_csv(
        os.path.join(output_path, "lightgbm_global_feature_ranking.txt"),
        index=False,
        sep='\t',
        float_format='%.4f'
    )

    print("\nReduced model feature importance:")
    print(importance_reduced.sort_values('importance', ascending=False))

    print("\nRunning reduced model SHAP analysis")
    shap_analyzer = ShapAnalyzer(reduced_model, X_train_reduced)
    shap_analyzer.analyze_global_importance(
        X_train_reduced,
        output_dir=os.path.join(top_30_dir, "global", "shap")
    )

    print("\nAnalyzing reduced model scenarios")
    analyze_random_red_light_scenario(
        X_train_reduced, X_test_reduced, scenario_test, y_test, reduced_model, test_pred_reduced, top_30_dir
    )
    analyze_random_side_move_scenario(
        X_train_reduced, X_test_reduced, scenario_test, y_test, reduced_model, test_pred_reduced, top_30_dir
    )
    analyze_random_normal_scenario(
        X_train_reduced, X_test_reduced, scenario_test, y_test, reduced_model, test_pred_reduced, top_30_dir
    )

if __name__ == "__main__":
    main()
