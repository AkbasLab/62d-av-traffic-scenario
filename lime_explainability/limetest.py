from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from lime.lime_tabular import LimeTabularExplainer
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class CollisionDataLoader:
    def __init__(self, params_path, scores_path):
        self.params_path = params_path
        self.scores_path = scores_path

    def load_data(self):
        param_df = pd.read_feather(self.params_path)
        scores_df = pd.read_feather(self.scores_path)
        num_collisions_df = pd.DataFrame({
            'num_collisions': scores_df['collisions'].apply(len)
        })
        full_df = pd.concat([param_df, num_collisions_df], axis=1)
        X = full_df.drop('num_collisions', axis=1)
        y = full_df['num_collisions']
        return self.split_data(X, y)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

class CollisionModel:
    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.07,
            depth=8,
            l2_leaf_reg=2,
            bagging_temperature=0.5,
            random_state=555,
            verbose=0
        )

    def cross_validate(self, X_train, y_train):
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        print("Cross-validation performance:")
        print(f"R^2: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def train_model(self, X_train,y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        test_pred = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        print("\nTest set performance:")
        print(f"MSE: {test_mse:.3f}")
        print(f"R^2: {test_r2:.3f}")

        print("\nPrediction statistics:")
        print("Actual values - Mean:", f"{y_test.mean():.3f}", "Std:", f"{y_test.std():.3f}")
        print("Predicted values - Mean:", f"{test_pred.mean():.3f}", "Std:", f"{test_pred.std():.3f}")

        return test_pred

    def predict(self, X):
        return self.model.predict(X)

class LIMEAnalyzer:
    def __init__(self, X_train, model):
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=['num_collisions'],
            mode='regression'
        )
        self.model = model

    def _get_cases(self, y_test, test_pred):
        random_idx = np.random.choice(y_test.index)
        high_collision_idx = y_test.idxmax()
        zero_collision_mask = (y_test == 0)
        zero_collision_idx = y_test[zero_collision_mask].index[0] if zero_collision_mask.any() else y_test.idxmin()
        prediction_errors = np.abs(y_test - test_pred)
        large_error_idx = prediction_errors.idxmax()

        return {
            'Random_Case' : random_idx,
            'High_Collision_Case': high_collision_idx,
            'Zero_Low_Collision_Case': zero_collision_idx,
            'Large_Error_Case': large_error_idx
        }

    def analyze_and_plot(self, X_test, y_test, test_pred, output_dir):
        cases = self._get_cases(y_test, test_pred)

        for case_name, idx in cases.items():
            print(f"\n{case_name.replace('_', ' ')}:")
            print(f"Actual collisions: {y_test.loc[idx]:.0f}")
            print(f"Predicted collisions: {test_pred[y_test.index.get_loc(idx)]:.2f}")

            exp = self.explainer.explain_instance(
                X_test.loc[idx].values,
                self.model.predict,
                num_features=10
            )

            print("\n\nTop contributing features:")
            for feature, impact in exp.as_list():
                print(f"{feature}: {impact:.4f}")

            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f"{case_name.replace('_', ' ')}\nActual: {y_test.loc[idx]:.0f}, "
                     f"Predicted: {test_pred[y_test.index.get_loc(idx)]:.2f}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{case_name}_plot.png"))
            plt.close()

def main():
    data_loader = CollisionDataLoader(
        "out/tuning-data/gamma_cross_eb_left_params.feather",
        "out/tuning-data/gamma_cross_eb_left_scores.feather"
    )

    X_train, X_test, y_train, y_test = data_loader.load_data()

    model = CollisionModel()
    model.cross_validate(X_train, y_train)
    model.train_model(X_train, y_train)
    test_pred = model.evaluate(X_test, y_test)

    lime_analyzer = LIMEAnalyzer(X_train, model)
    lime_analyzer.analyze_and_plot(X_test, y_test, test_pred, "out")

if __name__ == "__main__":
    main()
