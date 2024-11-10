from sklearn.model_selection import train_test_split
import pandas as pd

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
