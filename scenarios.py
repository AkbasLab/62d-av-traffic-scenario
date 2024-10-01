import scenarioxp as sxp
import pandas as pd

import constants
import utils
import traci

class GammaCrossAI:
    def __init__(self):
        self.net = utils.parse_net(constants.traci.gamma_cross.net_file)
        self.sidVehicle = {} # vehicles that want to do side move
        return
    
    def on_step(self):
        # step2 of side move
        for v in self.sidVehicle:
            traci.vehicle.moveTo(v,self.sidVehicle[v][0],self.sidVehicle[v][1]+8)
        self.sidVehicle.clear()

        for e in self.net:
            vehicles = traci.edge.getLastStepVehicleIDs(e)
            # change the Aggressive vehicles' speed mode---break traffic light
            for v in vehicles:
                lane = traci.vehicle.getLaneID(str(v))
                Id = traci.vehicle.getTypeID(v)
                if Id == "AggrCar":
                    traci.vehicle.setSpeedMode(v,7)
            
            # step1 of side move
            for l in self.net[e]:
                length = traci.lane.getLength(l)
                if traci.lane.getLastStepHaltingNumber(l) >= 2:
                    for v1 in traci.lane.getLastStepVehicleIDs(l):
                        pos = traci.vehicle.getLanePosition(str(v1))
                        if traci.vehicle.getTypeID(v1) == "AggrCar" and length - pos < 12 and length - pos > 3:
                            b = int(l[4]) # get the index of the lane
                            for l1 in self.net[e]:
                                a = int(l1[4])
                                if l1 != l and abs(b-a) == 1 and traci.lane.getLastStepHaltingNumber(l1) == 0:
                                    traci.vehicle.highlight(v1, (255, 0, 0, 255), -1, 1, 4,0)
                                    traci.vehicle.moveTo(v1,l1,pos+8)
                                    self.sidVehicle[v1] = (l, pos+8)
                                    break
        return

class GammaCrossScenario(sxp.Scenario):
    def __init__(self, params : pd.Series):
        traci.simulation.loadState(constants.sumo.init_state_file)
        
        self._params = params
        ai = GammaCrossAI()
        
        self._score = pd.Series({
            "collision" : 0
        })

        if constants.sumo.gui:
            traci.gui.setZoom(
                constants.sumo.default_view, 
                constants.sumo.dut_zoom
            )

        self.idle_until_start_time()
        self.add_vehicles()        

        # return
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            ai.on_step()

            # Dut complete
            if "o" in traci.vehicle.getLaneID(constants.DUT) \
                and traci.vehicle.getLanePosition(constants.DUT) > 20:
                break
            continue
        return
    
    @property
    def score(self) -> pd.Series:
        return self._score
    
    @property
    def params(self) -> pd.Series:
        return self._params

    def idle_until_start_time(self):
        start_time = self.params["time0"]
        while traci.simulation.getTime() < start_time:
            traci.simulationStep()
        return

    def add_vehicles(self):

        # Prepare the DUT
        traci.vehicle.add(
            constants.DUT,
            "warmup",
            typeID= "AggrCar",
            departLane = 60,
            departSpeed = utils.kph2mps(self.params["dut_s0"])
        )
        traci.vehicle.setColor(constants.DUT, constants.RGBA.light_blue)
        traci.vehicle.setLaneChangeMode(constants.DUT,0)

        # Add traffic
        self.add_traffic()

        #  Move the DUT
        directions = {
            "eb" : 1,
            "wb" : 2,
            "nb" : 3,
            "sb" : 4
        }
        rid = constants.traci.gamma_cross.dut_route
        direction = directions[rid[:2]]

        """
        Move DUT to new route
        20m behind the 5th car spot in the center lane.
        """
        turn_lane_length = constants.traci.gamma_cross.turn_lane_length
        pos = turn_lane_length - 20 - 5*7 - 2 - 20
        lid = "%dsi_1" % direction
        traci.vehicle.moveTo(
            constants.DUT,
            lid,
            pos
        )
        
        # Restore route
        traci.vehicle.setRouteID(constants.DUT, rid)

        # Restore lane change properties
        traci.vehicle.setLaneChangeMode(
            constants.DUT,
            constants.traci.default_lane_change_behavior
        )

        # Focus on DUT
        if constants.sumo.gui:
            traci.gui.trackVehicle(constants.sumo.default_view, constants.DUT)
            traci.gui.setZoom(
                constants.sumo.default_view, 
                constants.sumo.dut_zoom
            )
        return

    def add_traffic(self):
        
        directions = {
            "eb" : 1,
            "wb" : 2,
            "nb" : 3,
            "sb" : 4
        }
        lanes = {
            "right" : 0, 
            "straight" : 1, 
            "left" : 2
        }
        vtypes = [None, "Car", "AggrCar"]

        """
        Traffic Vehicles
        """
        # Prepare vehicle data
        vehicle_data = []
        for i in range(1,5+1):
            for dir in directions.keys():
                for lane in lanes.keys():
                    # Vehicle Type
                    i_vtype = self.params["vtype_%s_%s%d" % (dir, lane, i)]
                    vtype = vtypes[int(i_vtype)]
                    if vtype is None:
                        continue
                    
                    # Speed
                    kph = self.params["%s_%s_s0" % (dir,lane)]
                    mps = utils.kph2mps(kph)

                    # Route ID
                    rid = "%s_%s" % (dir, lane)

                    # Starting Lane
                    lid = "%dsi_%d" % (directions[dir], lanes[lane])

                    # Vehicle ID
                    vid = "%s_%s%d" % (dir, lane, i)
                    
                    s = pd.Series ({
                        "vtype" : vtype,
                        "s0" : mps,
                        "rid" : rid,
                        "lid" : lid,
                        "vid" : vid
                    })
                    vehicle_data.append(s)
                    continue
                continue
            continue
        df = pd.DataFrame(vehicle_data)

        # Put on warmup edge
        for i in range(len(df.index)):
            s = df.iloc[i]

            traci.vehicle.add(
                s["vid"],
                routeID = "warmup", 
                departSpeed = s["s0"],
                departLane = i,
                typeID = s["vtype"]
            )

            # Disable lane change
            traci.vehicle.setLaneChangeMode(s["vid"],0)
            continue

        # Add vehicles to simulation
        traci.simulationStep()

        # Move to the correct edges
        for i in range(len(df.index)):
            s = df.iloc[i]
            
            ipos = int(s["vid"][-1])
            lane_length = constants.traci.gamma_cross.turn_lane_length
            pos = lane_length - 20 - (ipos-1)*(5+2)
            traci.vehicle.moveTo(
                s["vid"], 
                s["lid"], 
                pos = pos
            )
            traci.vehicle.setRouteID(s["vid"], s["rid"])
            traci.vehicle.setLaneChangeMode(
                s["vid"],
                constants.traci.default_lane_change_behavior    
            )
            continue
        return
