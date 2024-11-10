from sklearn.model_selection import train_test_split
import pandas as pd

class CollisionDataLoader:
    def __init__(self, params_path: str, scores_path: str):
        self.params_path = params_path
        self.scores_path = scores_path
        self.params_df = pd.read_feather(self.params_path)
        self.scores_df = pd.read_feather(self.scores_path)

    def create_new_column_from_length(self, column_to_use: str, new_column: str):
        num_collisions_df = pd.DataFrame({
            new_column: self.scores_df[column_to_use].apply(len)
        })
        full_df = pd.concat([self.params_df, num_collisions_df], axis=1)
        return full_df

    def split_data(self, input_df: pd.DataFrame, column_as_y: str | None = None, test_size: float = 0.2, random_state: int | None = None):
        X = input_df.drop(column_as_y, axis=1)
        y = input_df[column_as_y]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
