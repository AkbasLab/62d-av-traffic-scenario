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
        
        tsc = lambda s : len(s["collisions"]) > 100
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
        prefix = "gamma_cross_%s" % constants.traci.gamma_cross.dut_route
        seq_exp.params_history.to_feather("out/%s_params.feather" % prefix)
        seq_exp.score_history.to_feather("out/%s_scores.feather" % prefix)

        return
    
    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    def random_seed(self):
        return self.rng.randint(2**32-1)

if __name__ == "__main__":
    Runner()