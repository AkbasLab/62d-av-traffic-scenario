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
        self.manager = sxp.ScenarioManager(df)
        
        self.traci_client = traci_clients.GenericClient(
            constants.traci.gamma_cross.config)
        
        self.scenario = scenarios.GammaCrossScenario

        
        self.monte_carlo()
        return



    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    def random_seed(self):
        return self.rng.randint(2**32-1)
    
    def monte_carlo(self):
        tsc = lambda s : len(s["collisions"]) > 100

        seq_exp = sxp.SequenceExplorer(
            strategy = sxp.SequenceExplorer.MONTE_CARLO,
            seed = self.random_seed(),
            scenario_manager = self.manager,
            scenario = self.scenario,
            target_score_classifier = tsc,
            scramble = False,
            fast_foward = self.random_seed() % 10000
        )

        for i in range(constants.n_tests):
            print("Test %d" % i, end="\r")
            seq_exp.step()
            # break
        
        self.traci_client.close()

        prefix = "gamma_cross_%s" % constants.traci.gamma_cross.dut_route
        seq_exp.params_history.to_feather("out/mc_%s_params.feather" % prefix)
        seq_exp.score_history.to_feather("out/mc_%s_scores.feather" % prefix)
        return

if __name__ == "__main__":
    Runner()