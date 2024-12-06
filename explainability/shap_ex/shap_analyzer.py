import matplotlib.pyplot as plt
import shap
import os
import time
import numpy as np

class ShapAnalyzer:
    def __init__(self, model, background_data):
        self.model = model
        # Convert background data to numpy array if it's a pandas DataFrame
        if hasattr(background_data, 'values'):
            background_data = background_data.values
        # Use TreeExplainer specifically for CatBoost
        self.explainer = shap.TreeExplainer(model)

    def analyze_specific_scenario(self, X, y, scenario_idx, output_dir):
        """Analyze a specific scenario using SHAP"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Time the SHAP value computation
        start_analysis = time.perf_counter()
        instance = X.loc[[scenario_idx]]
        if hasattr(instance, 'values'):
            instance = instance.values
        shap_values = self.explainer.shap_values(instance)
        analysis_time = time.perf_counter() - start_analysis

        # Time the plotting
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
        plt.title(f"SHAP Values for Scenario {scenario_idx}")
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
            # Analysis time
            start_analysis = time.perf_counter()
            instance = X.loc[[idx]]
            if hasattr(instance, 'values'):
                instance = instance.values
            shap_values = self.explainer.shap_values(instance)
            analysis_time = time.perf_counter() - start_analysis
            analysis_total += analysis_time

            # Plot time
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
            # Analysis time
            start_analysis = time.perf_counter()
            instance = X.loc[[idx]]
            if hasattr(instance, 'values'):
                instance = instance.values
            shap_values = self.explainer.shap_values(instance)
            analysis_time = time.perf_counter() - start_analysis
            analysis_total += analysis_time

            # Plot time
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
