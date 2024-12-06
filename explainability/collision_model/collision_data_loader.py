from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class CollisionDataLoader:
    def __init__(self, params_path: str, scores_path: str):
        self.params_path = params_path
        self.scores_path = scores_path
        self.params_df = pd.read_feather(self.params_path)
        self.scores_df = pd.read_feather(self.scores_path)

        if len(self.params_df) != len(self.scores_df):
            raise ValueError(f"Params ({len(self.params_df)} rows) and Scores ({len(self.scores_df)} rows) must have same length")

    def get_scenario_columns(self):
        return pd.DataFrame({
            'run_red_light': self.scores_df['run red light'],
            'side_move': self.scores_df['side move'].apply(lambda x: x >= 0)
        })

    def get_num_collisions(self, column_to_use: str = "collisions"):
        return self.scores_df[column_to_use].apply(len)

    @staticmethod
    def combine_datasets_with_movement(dataset_paths: list[tuple[str, str, str]],
                                     column_to_use: str = "collisions") -> pd.DataFrame:
        full_df = pd.DataFrame()

        for params_path, scores_path, movement_type in dataset_paths:
            loader = CollisionDataLoader(params_path, scores_path)

            temp_df = loader.params_df.copy()
            temp_df['movement_type'] = movement_type

            scenarios = loader.get_scenario_columns()
            temp_df['run_red_light'] = scenarios['run_red_light']
            temp_df['side_move'] = scenarios['side_move']

            temp_df['num_collisions'] = loader.get_num_collisions(column_to_use)

            full_df = pd.concat([full_df, temp_df], axis=0, ignore_index=True)

        # One-hot encode movement type
        movement_dummies = pd.get_dummies(full_df['movement_type'], prefix='movement')

        # Create final dataframe
        final_df = pd.concat([
            full_df.drop('movement_type', axis=1),  # Original data without movement_type
            movement_dummies,                        # One hot encoded movement types
        ], axis=1)

        # Shuffle
        return final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    @staticmethod
    def split_data(full_df: pd.DataFrame, test_size: float = 0.2, random_state: int | None = None):
        train_indices, test_indices = train_test_split(
            full_df.index,
            test_size=test_size,
            random_state=random_state
        )

        exclude_columns = ['num_collisions', 'run_red_light', 'side_move']
        feature_columns = [col for col in full_df.columns if col not in exclude_columns]

        X_train = full_df.loc[train_indices, feature_columns]
        X_test = full_df.loc[test_indices, feature_columns]
        y_train = full_df.loc[train_indices, 'num_collisions']
        y_test = full_df.loc[test_indices, 'num_collisions']

        scenario_train = full_df.loc[train_indices, ['run_red_light', 'side_move']]
        scenario_test = full_df.loc[test_indices, ['run_red_light', 'side_move']]

        assert len(X_train) == len(y_train) == len(scenario_train)
        assert len(X_test) == len(y_test) == len(scenario_test)
        assert all(X_train.index == scenario_train.index)
        assert all(X_test.index == scenario_test.index)

        return X_train, X_test, y_train, y_test, scenario_train, scenario_test
