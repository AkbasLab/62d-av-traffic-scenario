from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import os.path
import time

class LimeAnalyzer:
    def __init__(self, X_train, model, y):
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=[y],
            mode='regression',
            random_state=None,
        )
        self.model = model

    def analyze_specific_scenario(self, X_test, y_test, test_pred, scenario_idx, output_dir, verbose=False):
        # Analyze a specific scenario by index
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_time = time.perf_counter()
        exp = self.explainer.explain_instance(
            X_test.loc[scenario_idx].values,
            self.model.predict,
            num_features=65,
            num_samples=5000
        )
        analysis_time = time.perf_counter() - start_time

        if verbose:
            print(f"\nAnalyzing scenario {scenario_idx}:")
            print(f"Actual collisions: {y_test.loc[scenario_idx]:.0f}")
            print(f"Predicted collisions: {test_pred[y_test.index.get_loc(scenario_idx)]:.2f}")
            print(f"Analysis took {analysis_time:.4f} seconds")
            print("\nTop contributing features:")
            for feature, impact in exp.as_list():
                print(f"{feature}: {impact:.4f}")

        output_path = os.path.join(output_dir, f"scenario_{scenario_idx}_analysis.html")
        exp.save_to_file(output_path)

    def analyze_red_light_cases(self, X_test, y_test, test_pred, output_dir, verbose=False):
        """Analyze all cases where run_red_light is True"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        red_light_cases = X_test[X_test['run_red_light'] == True].index
        total_start_time = time.perf_counter()
        total_analysis_time = 0

        if verbose:
            print(f"\nAnalyzing {len(red_light_cases)} red light cases")

        for idx in red_light_cases:
            start_time = time.perf_counter()
            exp = self.explainer.explain_instance(
                X_test.loc[idx].values,
                self.model.predict,
                num_features=65,
                num_samples=5000
            )
            analysis_time = time.perf_counter() - start_time
            total_analysis_time += analysis_time

            if verbose:
                print(f"\nCase {idx}:")
                print(f"Analysis took {analysis_time:.4f} seconds")

            output_path = os.path.join(output_dir, f"red_light_case_{idx}.html")
            exp.save_to_file(output_path)

        total_time = time.perf_counter() - total_start_time
        if verbose:
            print(f"\nTotal analysis time: {total_time:.4f} seconds")
            print(f"Average time per case: {total_time/len(red_light_cases):.4f} seconds")
            print(f"Pure analysis time: {total_analysis_time:.4f} seconds")

    def analyze_side_move_cases(self, X_test, y_test, test_pred, output_dir, verbose=False):
        """Analyze all cases where side_move is True"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        side_move_cases = X_test[X_test['side_move'] == True].index
        total_start_time = time.perf_counter()
        total_analysis_time = 0

        if verbose:
            print(f"\nAnalyzing {len(side_move_cases)} side move cases")

        for idx in side_move_cases:
            start_time = time.perf_counter()
            exp = self.explainer.explain_instance(
                X_test.loc[idx].values,
                self.model.predict,
                num_features=65,
                num_samples=5000
            )
            analysis_time = time.perf_counter() - start_time
            total_analysis_time += analysis_time

            if verbose:
                print(f"\nCase {idx}:")
                print(f"Analysis took {analysis_time:.4f} seconds")

            output_path = os.path.join(output_dir, f"side_move_case_{idx}.html")
            exp.save_to_file(output_path)

        total_time = time.perf_counter() - total_start_time
        if verbose:
            print(f"\nTotal analysis time: {total_time:.4f} seconds")
            print(f"Average time per case: {total_time/len(side_move_cases):.4f} seconds")
            print(f"Pure analysis time: {total_analysis_time:.4f} seconds")
