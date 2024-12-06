from collision_model import CollisionDataLoader, CollisionModel
from lime_ex import LimeAnalyzer
from shap_ex import ShapAnalyzer

EXAMPLE_PARAM_DATA = "out/mc/mc_gamma_cross_eb_left_params.feather"
EXAMPLE_SCORE_DATA = "out/mc/mc_gamma_cross_eb_left_scores.feather"
OUTPUT_DIR = "out/explainability"
Y_COLUMN = "num_collisions"
RANDOM_STATE = None

def main():
    data_loader = CollisionDataLoader(
        EXAMPLE_PARAM_DATA,
        EXAMPLE_SCORE_DATA
    )

    data_df = data_loader.create_new_column_from_length(column_to_use="collisions", new_column=Y_COLUMN)

    X_train, X_test, y_train, y_test = data_loader.split_data(input_df=data_df, column_as_y=Y_COLUMN, random_state=RANDOM_STATE)

    model = CollisionModel(random_state=RANDOM_STATE)
    model.cross_validate(X_train, y_train)
    model.train_model(X_train, y_train)
    test_pred = model.evaluate(X_test, y_test)

    lime_analyzer = LimeAnalyzer(X_train, model, y=Y_COLUMN)
    lime_analyzer.analyze_and_plot(X_test, y_test, test_pred, OUTPUT_DIR + "/lime", verbose=True)

    shap_analyzer = ShapAnalyzer(model.model, X_train)
    shap_analyzer.analyze_and_plot(X_test, y_test, output_dir=OUTPUT_DIR + "/shap")

if __name__ == "__main__":
    main()
