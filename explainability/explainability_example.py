
def main():
    data_loader = CollisionDataLoader(
        "out/mc/mc_gamma_cross_eb_left_params.feather",
        "out/mc/mc_gamma_cross_eb_left_scores.feather"
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
