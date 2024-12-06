from collision_model import CollisionDataLoader
from lime_ex import LimeAnalyzer
from shap_ex import ShapAnalyzer
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

class BestScoreCallback:
    def __init__(self):
        self.best_score = float('inf')
        self.best_iteration = 0

    def __call__(self, env):
        score = env.evaluation_result_list[0][2]
        iteration = env.iteration

        if score < self.best_score:
            self.best_score = score
            self.best_iteration = iteration
            print(f"\nNew best score at iteration {iteration}: {score:.4f}")

def evaluate_model_performance(model, X_test, y_test):
    """Print model performance metrics"""
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
    """Analyze one random red light running case with both LIME and SHAP"""
    print("\n=== Analyzing Random Red Light Running Case ===")

    red_light_cases = scenario_test[scenario_test['run_red_light'] == True].index
    if len(red_light_cases) == 0:
        print("No red light cases found in test set")
        return

    random_idx = np.random.choice(red_light_cases)
    print(f"Selected red light scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "red_light")

def analyze_random_side_move_scenario(X_train, X_test, scenario_test, y_test, model, test_pred):
    """Analyze one random side move case with both LIME and SHAP"""
    print("\n=== Analyzing Random Side Move Case ===")

    side_move_cases = scenario_test[scenario_test['side_move'] == True].index
    if len(side_move_cases) == 0:
        print("No side move cases found in test set")
        return

    random_idx = np.random.choice(side_move_cases)
    print(f"Selected side move scenario: {random_idx}")

    analyze_scenario(X_train, X_test, y_test, test_pred, model, random_idx, "side_move")

def analyze_random_normal_scenario(X_train, X_test, scenario_test, y_test, model, test_pred):
    """Analyze one random normal case"""
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
    """Analyze a specific scenario with both LIME and SHAP"""
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
    full_df = CollisionDataLoader.combine_datasets(DATASETS, add_movement_vars=True)

    print(f"Dataset shape: {full_df.shape}")
    print("\nFeature names:")
    feature_cols = [col for col in full_df.columns if col not in ['num_collisions', 'run_red_light', 'side_move']]
    print('\n'.join(feature_cols))

    X_train, X_test, y_train, y_test, scenario_train, scenario_test = CollisionDataLoader.split_data(
        full_df,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    param_distributions = {
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'num_leaves': randint(20, 100),
        'min_child_samples': randint(10, 50),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 2),
        'reg_lambda': uniform(0, 2),
        'min_split_gain': uniform(0, 1),
        'boosting_type': ['gbdt', 'dart']
    }

    random_search = RandomizedSearchCV(
        estimator=LGBMRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective='regression',
            metric='rmse',
            verbosity=-1
        ),
        param_distributions=param_distributions,
        n_iter=100,
        cv=5,
        scoring='r2',
        random_state=RANDOM_STATE,
        verbose=2
    )

    print("\nStarting Random Search...")
    random_search.fit(X_train_search, y_train_search)

    print("\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best R² score: {random_search.best_score_:.4f}")

    # Save random search results
    search_results = pd.DataFrame(random_search.cv_results_)
    os.makedirs(os.path.join(OUTPUT_DIR, "lightgbm", "tuning"), exist_ok=True)
    search_results.to_csv(
        os.path.join(OUTPUT_DIR, "lightgbm", "tuning", "random_search_results.csv"),
        index=False
    )

    # Train final model with best parameters and early stopping (early stopping doesnt work atm, check warning)
    print("\nTraining final model with best parameters...")
    best_model = LGBMRegressor(
        **random_search.best_params_,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    best_score_tracker = BestScoreCallback()
    best_model.fit(
        X_train,
        y_train,
        eval_metric='rmse',
        eval_set=[(X_test, y_test)],
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=50),
            lightgbm.log_evaluation(period=100),
            best_score_tracker
        ],
    )

    print(f"\nBest score achieved: {best_score_tracker.best_score:.4f}")
    print(f"At iteration: {best_score_tracker.best_iteration}")

    test_pred = evaluate_model_performance(best_model, X_test, y_test)

    #lightgbm default features
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    })
    output_path = os.path.join(OUTPUT_DIR, "lightgbm", "global")
    os.makedirs(output_path, exist_ok=True)
    importance.to_csv(
        os.path.join(output_path, "lightgbm_global_feature_ranking.txt"),
        index=False,
        sep='\t',
        float_format='%.4f'
    )

    print("\nTop 10 important features:")
    print(importance.sort_values('importance', ascending=False).head(10))

    # global shap first
    shap_analyzer = ShapAnalyzer(best_model, X_train)
    shap_analyzer.analyze_global_importance(
        X_train,
        output_dir=f"{OUTPUT_DIR}/shap/global"
    )

    # then shap/lime scenarios
    analyze_random_red_light_scenario(X_train, X_test, scenario_test, y_test, best_model, test_pred)
    analyze_random_side_move_scenario(X_train, X_test, scenario_test, y_test, best_model, test_pred)
    analyze_random_normal_scenario(X_train, X_test, scenario_test, y_test, best_model, test_pred)

if __name__ == "__main__":
    main()
