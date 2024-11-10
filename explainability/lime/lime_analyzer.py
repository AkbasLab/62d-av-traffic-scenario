
from lime.lime_tabular import LimeTabularExplainer


class LimeAnalyzer:
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
