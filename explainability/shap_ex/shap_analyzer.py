import matplotlib.pyplot as plt
import shap
import os

class ShapAnalyzer:
    def __init__(self, model, background_data):
        self.model = model
        self.explainer = shap.Explainer(model, background_data)

    def analyze_and_plot(self, X, y, output_dir, waterfall: bool = True, beeswarm: bool = True, barplot: bool = True):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            shap_values = self.explainer(X)

            if waterfall:
                plt.figure()
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "waterfall_plot.png"), bbox_inches='tight', dpi=300)
                plt.close()
            if beeswarm:
                plt.figure()
                shap.plots.beeswarm(shap_values, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "beeswarm_plot.png"), bbox_inches='tight', dpi=300)
                plt.close()
            if barplot:
                plt.figure()
                shap.plots.bar(shap_values, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "bar_plot.png"), bbox_inches='tight', dpi=300)
                plt.close()
