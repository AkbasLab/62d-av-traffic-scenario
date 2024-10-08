import constants
import traci_clients
import scenarios

import scenarioxp as sxp
import pandas as pd
import numpy as np


class Runner:
    def __init__(self):
        self._rng = np.random.RandomState(seed=constants.seed)

        # Build the Manager
        fn = "scenario_config/cross-gama-params.xlsx"
        df = pd.read_excel(
            fn, 
            engine = 'openpyxl',
            usecols = ["feat","min","max","inc"],
        )
        manager = sxp.ScenarioManager(df)
        
        traci_client = traci_clients.GenericClient(
            constants.traci.gamma_cross.config)
        
        tsc = lambda s : s["collision"] > 0
        scenario = scenarios.GammaCrossScenario

        seq_exp = sxp.SequenceExplorer(
            strategy = sxp.SequenceExplorer.MONTE_CARLO,
            seed = self.random_seed(),
            scenario_manager = manager,
            scenario = scenario,
            target_score_classifier = tsc,
            scramble = False,
            fast_foward = self.random_seed() % 10000
        )

        for i in range(constants.n_tests):
            print("Test %d" % i, end="\r")
            seq_exp.step()
            # break
        

        traci_client.close()
        return
    
    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    def random_seed(self):
        return self.rng.randint(2**32-1)

if __name__ == "__main__":
    Runner()