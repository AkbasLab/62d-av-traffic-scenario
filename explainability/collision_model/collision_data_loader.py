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
    def combine_datasets(dataset_paths: list[tuple[str, str, str]],
                        column_to_use: str = "collisions",
                        add_movement_vars: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        model_df = pd.DataFrame()
        scenario_df = pd.DataFrame()

        for params_path, scores_path, movement_type in dataset_paths:
            loader = CollisionDataLoader(params_path, scores_path)
            temp_df = loader.params_df.copy()

            # Add movement vars to model features
            if add_movement_vars:
                temp_df['movement_type'] = movement_type

            # Keep scenarios separate
            temp_scenarios = loader.get_scenario_columns()

            # Add num_collisions to both
            temp_df['num_collisions'] = loader.get_num_collisions(column_to_use)
            temp_scenarios['num_collisions'] = loader.get_num_collisions(column_to_use)

            model_df = pd.concat([model_df, temp_df], axis=0, ignore_index=True)
            scenario_df = pd.concat([scenario_df, temp_scenarios], axis=0, ignore_index=True)

        if add_movement_vars:
            movement_dummies = pd.get_dummies(model_df['movement_type'], prefix='movement')
            model_df = pd.concat([
                model_df.drop('movement_type', axis=1),
                movement_dummies
            ], axis=1)

        # Use same shuffle order for both
        shuffle_idx = np.random.permutation(len(model_df))
        return model_df.iloc[shuffle_idx].reset_index(drop=True), scenario_df.iloc[shuffle_idx].reset_index(drop=True)
