import matplotlib.pyplot as plt
import shap
import os
import time
import numpy as np
import pandas as pd

class ShapAnalyzer:
    def __init__(self, model, background_data):
        self.model = model
        if hasattr(background_data, 'values'):
            background_data = background_data.values
        self.explainer = shap.TreeExplainer(model)

    def analyze_global_importance(self, X, output_dir):
        """Create global SHAP importance plots (bar and beeswarm)"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("\nCalculating global SHAP values...")
        start_time = time.perf_counter()

        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = X

        shap_values = self.explainer.shap_values(X_values)
        mean_abs_shap = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_abs_shap
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
        feature_importance_sorted.to_csv(
            os.path.join(output_dir, "global_shap.txt"),
            index=False,
            sep='\t',
            float_format='%.4f'
        )
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False
        )
        plt.title("Global SHAP Feature Importance (Bar Plot)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "global_shap_bar.pdf"),
                    bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="dot", # beeswarm
            show=False
        )
        plt.title("Global SHAP Feature Importance (Beeswarm Plot)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "global_shap_beeswarm.pdf"),
                    bbox_inches='tight', dpi=300)
        plt.close()

        analysis_time = time.perf_counter() - start_time
        print(f"Global SHAP analysis took {analysis_time:.4f} seconds")

    def analyze_specific_scenario(self, X, y, scenario_idx, output_dir, actual_value=None, predicted_value=None):
        """Analyze a specific scenario using SHAP"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # time shap completion
        start_analysis = time.perf_counter()
        instance = X.loc[[scenario_idx]]
        if hasattr(instance, 'values'):
            instance = instance.values
        shap_values = self.explainer.shap_values(instance)
        analysis_time = time.perf_counter() - start_analysis

        # Create and save local feature importance rankings
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(shap_values[0])  # Take absolute values for importance
        })
        feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
        feature_importance_sorted.to_csv(
            os.path.join(output_dir, "local_shap.txt"),
            index=False,
            sep='\t',
            float_format='%.4f'
        )

        # time spent plotting
        start_plot = time.perf_counter()
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=X.columns
            ),
            show=False
        )

        title = "Local SHAP Feature Importance\n"
        if actual_value is not None:
            title += f"Number of collisions: {actual_value}\n"
        if predicted_value is not None:
            title += f"Predicted number of collisions: {predicted_value:.2f}"

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scenario_{scenario_idx}_waterfall.pdf"),
                    bbox_inches='tight', dpi=300)
        plt.close()
        plot_time = time.perf_counter() - start_plot

        print(f"SHAP analysis for scenario {scenario_idx} took {analysis_time + plot_time:.4f} seconds")
        print(f"(Analysis: {analysis_time:.4f}s, Plotting: {plot_time:.4f}s)")

    def analyze_red_light_cases(self, X, y, output_dir):
        """Analyze all red light cases using SHAP"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        red_light_cases = X[X['run_red_light'] == True]
        total_start = time.perf_counter()
        analysis_total = 0
        plot_total = 0

        print(f"\nAnalyzing {len(red_light_cases)} red light cases")

        for idx in red_light_cases.index:
            start_analysis = time.perf_counter()
            instance = X.loc[[idx]]
            if hasattr(instance, 'values'):
                instance = instance.values
            shap_values = self.explainer.shap_values(instance)
            analysis_time = time.perf_counter() - start_analysis
            analysis_total += analysis_time

            start_plot = time.perf_counter()
            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value,
                    data=instance[0],
                    feature_names=X.columns
                ),
                show=False
            )
            plt.title(f"SHAP Values for Red Light Case {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"red_light_case_{idx}_waterfall.pdf"),
                       bbox_inches='tight', dpi=300)
            plt.close()
            plot_time = time.perf_counter() - start_plot
            plot_total += plot_time

        total_time = time.perf_counter() - total_start
        num_cases = len(red_light_cases)

        print(f"\nSHAP analysis for {num_cases} red light cases took {total_time:.4f} seconds")
        print(f"Average time per case: {total_time/num_cases:.4f} seconds")
        print(f"Total analysis time: {analysis_total:.4f} seconds")
        print(f"Total plotting time: {plot_total:.4f} seconds")

    def analyze_side_move_cases(self, X, y, output_dir):
        """Analyze all side move cases using SHAP"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        side_move_cases = X[X['side_move'] == True]
        total_start = time.perf_counter()
        analysis_total = 0
        plot_total = 0

        print(f"\nAnalyzing {len(side_move_cases)} side move cases")

        for idx in side_move_cases.index:
            start_analysis = time.perf_counter()
            instance = X.loc[[idx]]
            if hasattr(instance, 'values'):
                instance = instance.values
            shap_values = self.explainer.shap_values(instance)
            analysis_time = time.perf_counter() - start_analysis
            analysis_total += analysis_time

            start_plot = time.perf_counter()
            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value,
                    data=instance[0],
                    feature_names=X.columns
                ),
                show=False
            )
            plt.title(f"SHAP Values for Side Move Case {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"side_move_case_{idx}_waterfall.pdf"),
                       bbox_inches='tight', dpi=300)
            plt.close()
            plot_time = time.perf_counter() - start_plot
            plot_total += plot_time

        total_time = time.perf_counter() - total_start
        num_cases = len(side_move_cases)

        print(f"\nSHAP analysis for {num_cases} side move cases took {total_time:.4f} seconds")
        print(f"Average time per case: {total_time/num_cases:.4f} seconds")
        print(f"Total analysis time: {analysis_total:.4f} seconds")
        print(f"Total plotting time: {plot_total:.4f} seconds")

    def analyze_specific_scenario_classification(self, X, y, scenario_idx, output_dir, actual_value=None, predicted_value=None):
        """Analyze a specific scenario using SHAP for classification tasks"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_analysis = time.perf_counter()
        instance = X.loc[[scenario_idx]]
        if hasattr(instance, 'values'):
            instance = instance.values
        shap_values = self.explainer.shap_values(instance)
        analysis_time = time.perf_counter() - start_analysis

        # Create and save local feature importance rankings
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(shap_values[0])  # Take absolute values for importance
        })
        feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
        feature_importance_sorted.to_csv(
            os.path.join(output_dir, "local_shap.txt"),
            index=False,
            sep='\t',
            float_format='%.4f'
        )

        start_plot = time.perf_counter()
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=X.columns
            ),
            show=False
        )

        title = "Local SHAP Feature Importance\n"
        if actual_value is not None:
            title += f"Run Red Light: {actual_value}\n"
        if predicted_value is not None:
            title += f"Predicted Value: {predicted_value}"

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scenario_{scenario_idx}_waterfall.pdf"),
                    bbox_inches='tight', dpi=300)
        plt.close()
        plot_time = time.perf_counter() - start_plot

        print(f"SHAP analysis for scenario {scenario_idx} took {analysis_time + plot_time:.4f} seconds")
        print(f"(Analysis: {analysis_time:.4f}s, Plotting: {plot_time:.4f}s)")

    def analyze_specific_scenario_sidemove(self, X, y, scenario_idx, output_dir, actual_value=None, predicted_value=None):
        """Analyze a specific scenario using SHAP for side move classification"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_analysis = time.perf_counter()
        instance = X.loc[[scenario_idx]]
        if hasattr(instance, 'values'):
            instance = instance.values
        shap_values = self.explainer.shap_values(instance)
        analysis_time = time.perf_counter() - start_analysis

        # Create and save local feature importance rankings
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(shap_values[0])  # Take absolute values for importance
        })
        feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
        feature_importance_sorted.to_csv(
            os.path.join(output_dir, "local_shap.txt"),
            index=False,
            sep='\t',
            float_format='%.4f'
        )

        start_plot = time.perf_counter()
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=X.columns
            ),
            show=False
        )

        title = "Local SHAP Feature Importance\n"
        if actual_value is not None:
            title += f"Side Move Occurred: {actual_value}\n"
        if predicted_value is not None:
            title += f"Predicted Value: {predicted_value}"

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scenario_{scenario_idx}_waterfall.pdf"),
                    bbox_inches='tight', dpi=300)
        plt.close()
        plot_time = time.perf_counter() - start_plot

        print(f"SHAP analysis for scenario {scenario_idx} took {analysis_time + plot_time:.4f} seconds")
        print(f"(Analysis: {analysis_time:.4f}s, Plotting: {plot_time:.4f}s)")
