from typing import Callable
import constants
import traci_clients
import scenarios
import utils

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
        self._manager = sxp.ScenarioManager(df)
        
        self._traci_client = traci_clients.GenericClient(
            constants.traci.gamma_cross.config)
        
        self._scenario = scenarios.GammaCrossScenario

        self._seq_exp_history = []
        self._fs_exp_history = []
        self._brrt_exp_history = []
        self._n_tests = 0

        # self.target_run_red_light()
        self.target_side_move()
        # self.monte_carlo()

        self.traci_client.close()

        return

    @property
    def manager(self) -> sxp.ScenarioManager:
        return self._manager
    
    @property
    def tsc(self) -> Callable[[pd.Series], bool]:
        return self._tsc

    @property
    def traci_client(self) -> traci_clients.GenericClient:
        return self._traci_client

    @property
    def scenario(self) -> scenarios.GammaCrossScenario:
        return self._scenario

    @property
    def seq_exp_history(self) -> list[sxp.SequenceExplorer]:
        return self._seq_exp_history

    @property
    def fs_exp_history(self) -> list[sxp.FindSurfaceExplorer]:
        return self._fs_exp_history

    @property
    def brrt_exp_history(self) -> list[sxp.BoundaryRRTExplorer]:
        return self._brrt_exp_history
    
    @property
    def n_tests(self) -> int:
        return self._n_tests
    
    @property
    def params_df(self) -> pd.DataFrame:
        return self._params_df
    
    @property
    def scores_df(self) -> pd.DataFrame:
        return self._scores_df

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

        type_map = {
            constants.vehicle_types.aggresive : "a",
            constants.vehicle_types.conservative : "c"
        }
        c = type_map[constants.traci.gamma_cross.dut_type]

        prefix = "gamma_cross_%s_%s" % (c , constants.traci.gamma_cross.dut_route)
        seq_exp.params_history.to_feather("out/mc_%s_params.feather" % prefix)
        seq_exp.score_history.to_feather("out/mc_%s_scores.feather" % prefix)

        return

    def target_run_red_light(self):
        self._tsc = lambda s : s["run red light"] != -1
        
        print()

        i = 0
        while True:      
            print("\n:: ENVELOPE %d ::\n" % i)
            i += 1 
            self.find_and_explore_one_envelope(
                n_boundary_samples = constants.n_boundary_samples
            )
            if self.n_tests >= constants.n_tests:
                break
            continue

        print()

        self.flatten_tests()

        pd.set_option('display.max_columns', None)
        print(self.scores_df)

        type_map = {
            constants.vehicle_types.aggresive : "a",
            constants.vehicle_types.conservative : "c"
        }
        c = type_map[constants.traci.gamma_cross.dut_type]
        prefix = "gamma_cross_%s_%s" % (c ,
            constants.traci.gamma_cross.dut_route)

        self.params_df.to_feather("out/run_red_light_%s_params.feather" % prefix)
        self.scores_df.to_feather("out/run_red_light_%s_scores.feather" % prefix)
        return

    def target_side_move(self):
        self._tsc = lambda s : s["side move"] != -1
        
        print()

        i = 0
        while True:      
            print("\n:: ENVELOPE %d ::\n" % i)
            i += 1 
            self.find_and_explore_one_envelope(
                n_boundary_samples = constants.n_boundary_samples
            )
            if self.n_tests >= constants.n_tests:
                break
            continue

        print()

        self.flatten_tests()

        pd.set_option('display.max_columns', None)
        print(self.scores_df)

        type_map = {
            constants.vehicle_types.aggresive : "a",
            constants.vehicle_types.conservative : "c"
        }
        c = type_map[constants.traci.gamma_cross.dut_type]
        prefix = "gamma_cross_%s_%s" % (c ,
            constants.traci.gamma_cross.dut_route)

        self.params_df.to_feather(
            "%s/side_move_%s_params.feather" % (constants.output_dir, prefix))
        self.scores_df.to_feather(
            "%s/side_move_%s_scores.feather" % (constants.output_dir, prefix))

        return
    

    def find_and_explore_one_envelope(self, 
            n_boundary_samples : int
        ):
        """
        Locates and explores 1 performance envelope.

        :: Parameters ::
            n_boundary_samples : int
                The number of boundary samples to explore for one performance 
                envelope.
        """
        # Common arguments
        kwargs = {
            "scenario_manager" : self.manager,
            "target_score_classifier" : self.tsc,
            "scenario" : self.scenario
        }

        # Locate a performance envelope
        seq_exp = sxp.SequenceExplorer(
            strategy = sxp.SequenceExplorer.HALTON,
            seed = self.random_seed(),
            fast_foward = self.random_seed() % 10000,
            **kwargs
        )
        
        n_tests = 0

        seq_steps = 0
        while seq_exp.stage != seq_exp.STAGE_EXPLORATION_COMPLETE:
            seq_exp.step()
            seq_steps += 1
            kept = len(seq_exp._arr_history)
            n_tests = self.n_tests + seq_steps
            skipped = seq_steps - kept
            print("                                                    ", end="\r")
            print("%d -> Locating Envelope: %d kept, %d skipped" \
                  % (n_tests,  kept, skipped), end="\r")
            # print()

            if n_tests >= constants.n_tests:
                self._n_tests = n_tests
                self._seq_exp_history.append(seq_exp)
                return     
            continue
        
        
        self._n_tests = n_tests
        self._seq_exp_history.append(seq_exp)

        # Find the surface of the envelope.
        fs_exp = sxp.FindSurfaceExplorer(
            root = seq_exp._arr_history[-1],
            seed = self.random_seed(),
            **kwargs
        )
        fs_steps = 0
        while fs_exp.stage != fs_exp.STAGE_EXPLORATION_COMPLETE:
            fs_exp.step()
            fs_steps += 1
            kept = len(fs_exp._arr_history)
            n_tests = self.n_tests + kept
            skipped = fs_steps - kept
            print("                                                    ", end="\r")
            print("%d -> Locating Surface: %d kept, %d skipped" \
                    % (n_tests,  kept, skipped), end="\r")
            # print()

            if n_tests >= constants.n_tests:
                self._n_tests = n_tests
                self._fs_exp_history.append(fs_exp)
                return     
            continue
        
        self._n_tests = n_tests
        self._fs_exp_history.append(fs_exp)



        # follow the boundary
        root = fs_exp._arr_history[-1]
        brrt_exp = sxp.BoundaryRRTExplorer(
            root = root,
            root_n = sxp.orthonormalize(root, fs_exp.v)[0],
            strategy = "e",
            **kwargs
        )
        
        brrt_steps = 0
        for i in range(n_boundary_samples):
        # while len(brrt_exp._arr_history) != n_boundary_samples:
            brrt_exp.step()
            brrt_steps += 1
            kept = len(brrt_exp._arr_history)
            n_tests = self.n_tests + kept
            skipped = brrt_steps - kept
            print("                                                    ", end="\r")
            print("%d -> Following Boundary: %d kept, %d skipped" \
                  % (n_tests, kept, skipped), end="\r")
            # print()
            
            if n_tests >= constants.n_tests:
                self._n_tests = n_tests
                self._brrt_exp_history.append(brrt_exp)
                return
            continue

        self._n_tests = n_tests
        self._brrt_exp_history.append(brrt_exp)
        

        return

    def flatten_tests(self):
        params = []
        scores = []

        for i in range(len(self.seq_exp_history)):
            
            exp_params = []
            exp_scores = []

            for stage_exp in [
                (self.seq_exp_history, "seq"),
                (self.fs_exp_history, "fs"),
                (self.brrt_exp_history, "brrt")
            ]:
                exp_history, stage = stage_exp
                try:
                    pdf = exp_history[i].params_history\
                        .assign(envelope_id = i)\
                        .assign(stage = stage)
                    sdf = exp_history[i].score_history\
                        .assign(envelope_id = i)\
                        .assign(stage = stage)
                    exp_params.append( pdf )
                    exp_scores.append( sdf )
                except IndexError:
                    break
                continue

            exp_params_df = pd.concat(exp_params)
            exp_scores_df = pd.concat(exp_scores)
            
            params.append(exp_params_df)
            scores.append(exp_scores_df)
            continue

        params_df = pd.concat(params).reset_index(drop=True)
        scores_df = pd.concat(scores).reset_index(drop=True)


        scores_df["is_target"] = scores_df.apply(
            lambda s: self.tsc(s), axis=1)
        
        self._params_df = params_df
        self._scores_df = scores_df
        return

if __name__ == "__main__":
    Runner()