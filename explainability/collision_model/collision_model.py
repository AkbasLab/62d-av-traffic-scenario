from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor

class CollisionModel:
    def __init__(self, random_state=None):
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.07,
            depth=8,
            l2_leaf_reg=2,
            bagging_temperature=0.5,
            random_state=random_state,
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
