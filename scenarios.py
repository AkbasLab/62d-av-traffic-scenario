import scenarioxp as sxp
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from typing import Tuple, List

import traci._simulation

import constants
import utils
import traci

class GammaCrossAI:
    def __init__(self):
        self.net = utils.parse_net(constants.traci.gamma_cross.net_file)
        self.sidVehicle = {} # vehicles that want to do side move
        return
    
    def on_step(self) -> bool:
        dut_perform_side_move = False

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
                                    
                                    # Added code to check if DUT performs side move.
                                    if v1 == constants.DUT:
                                        dut_perform_side_move = True
                                    break
        return dut_perform_side_move




class GammaCrossScenario(sxp.Scenario):
    def __init__(self, params : pd.Series):
        traci.simulation.loadState(constants.sumo.init_state_file)
        
        self._params = params
        ai = GammaCrossAI()
        
        self._score = pd.Series({
            "collisions" : [],
            "speed (on enter)" : -1,
            "braking force" : 0,
            "braking force (norm)" : 0,
            "dtc (front)" : 9999,
            "ttc (front)" : 9999,
            "dtc (inter)" : 9999,
            "dtc (approach)" : 9999,
            "tl state (on enter)" : "",
            "foes in inter (on enter)" : [],
            "time (on enter)" : -1,
            "time (end)" : -1,
            "n stops" : 0,
            "side move" : -1,
            "run red light" : False
        })

        if constants.sumo.gui:
            traci.gui.setZoom(
                constants.sumo.default_view, 
                constants.sumo.dut_zoom
            )

        self.idle_until_start_time()
        self.add_vehicles()
        self.clear_polygons()
        self.add_passenger_polygons()

        self._start_time = traci.simulation.getTime()
        self._dut_speed_history = []

        if constants.sumo.pause_after_initialze:
            input()

        """
        Simulation Loop
        """
        prev_dut_lane_id = None
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Exit if DUT doesn't exist.
            if not constants.DUT in traci.vehicle.getIDList():
                break

            # AI Logic
            dut_perform_side_move = ai.on_step()
            if dut_perform_side_move:
                self.score["side move"] = self.get_time()

            # Metrics
            self.collision_metrics()                
            self.check_for_new_stops()
            self.foe_in_front_metrics()
            self.braking_force_metrics()

            # Find moment of entering/exiting intersection
            dut_lane_id = traci.vehicle.getLaneID(constants.DUT)
            if dut_lane_id[0] == "1":
                self.dut_approach()
            elif prev_dut_lane_id is None:
                pass
            elif prev_dut_lane_id[0] != ":" and dut_lane_id[0] == ":":
                self.dut_enter_intersection()
            elif prev_dut_lane_id[0] == ":" and dut_lane_id[0] != ":":
                self.dut_exit_intersection()

            # Logic within intersection
            if dut_lane_id[0] == ":":
                self.dut_isin_intersection()
        

            # Dut complete
            if "o" in traci.vehicle.getLaneID(constants.DUT) \
                and traci.vehicle.getLanePosition(constants.DUT) > 20:
                break

            prev_dut_lane_id = traci.vehicle.getLaneID(constants.DUT)
            continue

        self.score["time (end)"] = self.get_time()

        return
    
    @property
    def score(self) -> pd.Series:
        return self._score
    
    @property
    def params(self) -> pd.Series:
        return self._params
    
    @property
    def start_time(self) -> float:
        """
        Sim time when the simulation begins (after initializtion) in seconds.
        """
        return self._start_time
    
    @property
    def dut_speed_history(self) -> list[float]:
        """
        DUT speed history (in mps).
        """
        return self._dut_speed_history

    def dut_approach(self):
        self.dtc_approach_metrics()
        return

    def dut_enter_intersection(self):
        traci.vehicle.setColor(
            constants.DUT,
            constants.RGBA.cyan
        )
        self.score["time (on enter)"] = self.get_time()
        self.score["speed (on enter)"] = traci.vehicle.getSpeed(constants.DUT)

        # TL State
        tl_id = traci.trafficlight.getIDList()[0]
        tl_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        self.score["tl state (on enter)"] = tl_state

        # Does DUT run the red light?
        i_tl = constants.traci.gamma_cross.tl_order[
            constants.traci.gamma_cross.dut_route
        ] 
        self.score["run red light"] = tl_state[i_tl] == "r"
        
        # Foes in intersection
        self.score["foes in inter (on enter)"] = self.get_foes_in_intersection()
        return

    def dut_isin_intersection(self):
        # Collect intesection metrics when moving.
        if traci.vehicle.getSpeed(constants.DUT) > 0:
            self.dtc_intersection_metrics()
        return

    def dut_exit_intersection(self):
        traci.vehicle.setColor(
            constants.DUT,
            constants.RGBA.light_blue
        )
        return
    
    def dtc_intersection_metrics(self):
        """
        Get the vehicles within the intersection
        """
        assert traci.vehicle.getLaneID(constants.DUT)[0] == ":"
        # print()

        foe_polygons = []
        for vid in traci.vehicle.getIDList():
            if vid == constants.DUT:
                continue
            lid = traci.vehicle.getLaneID(vid)
            if lid[0] == ":":
                poly = Polygon( traci.polygon.getShape(vid) )
                foe_polygons.append(poly)
            continue  
        
        if len(foe_polygons) == 0:
            return 
        
        # Measure the distance from each foe to the DUT
        dut_polygon = Polygon( traci.polygon.getShape(constants.DUT) )
        dist = min([dut_polygon.distance(poly) for poly in foe_polygons])
        
        self.score["dtc (inter)"] = min(self.score["dtc (inter)"],dist)
        return
    
    def dtc_approach_metrics(self):
        """
        Get vehicles within the approach OR the intersection
        """
        assert traci.vehicle.getLaneID(constants.DUT)[0] == "1"

        foe_polygons = []
        for vid in traci.vehicle.getIDList():
            if vid == constants.DUT:
                continue
            lid = traci.vehicle.getLaneID(vid)
            if lid[0] in ":1":
                poly = Polygon( traci.polygon.getShape(vid) )
                foe_polygons.append(poly)
            continue    
        
        if len(foe_polygons) == 0:
            return 
        
        # Measure the distance from each foe to the DUT
        dut_polygon = Polygon( traci.polygon.getShape(constants.DUT) )
        dist = min([dut_polygon.distance(poly) for poly in foe_polygons])
        
        self.score["dtc (approach)"] = min(self.score["dtc (approach)"],dist)
        return

    def braking_force_metrics(self):
        accel = traci.vehicle.getAcceleration(constants.DUT)
        if accel >= 0:
            return
        
        brake = -accel
        if brake > self.score["braking force"]:
            self.score["braking force"] = brake

            decel = traci.vehicle.getDecel(constants.DUT)
            e_decel = traci.vehicle.getEmergencyDecel(constants.DUT)
            
            if brake <= decel:
                brake_norm = brake/decel
            else:
                brake_norm = 1 + (brake - decel)/(e_decel - decel)  

            # Correct for float rounding error at the upper limit
            self.score["braking force (norm)"] = brake_norm
        return

    def foe_in_front_metrics(self):
        foe = self.find_vehicle_in_front_of_dut()
        if foe is None:
            return
        
        # print("\n\n")

        # Distance to collission from shortest point on polygon
        foe_poly = Polygon( traci.polygon.getShape(foe) )
        dut_poly = Polygon( traci.polygon.getShape(constants.DUT) )
        dtc = dut_poly.distance(foe_poly)
        
        self.score["dtc (front)"] = min(self.score["dtc (front)"], dtc)

        # Time to collision
        foe_speed = traci.vehicle.getSpeed(foe)
        dut_speed = traci.vehicle.getSpeed(constants.DUT)
        rel_speed = dut_speed - foe_speed
        if rel_speed > 0:
            ttc = dtc / rel_speed 
            self.score["ttc (front)"] = min(self.score["ttc (front)"], ttc)
        return
    
    def find_vehicle_in_front_of_dut(self) -> str:
        """
        Finds the nearest vehicle in front of the DUT within the same lane.
        
        Returns vehicle ID or None.
        """
        # Get foes in front of DUT
        dut_lane = traci.vehicle.getLaneID(constants.DUT)

        # No Other vehicles in lane
        if traci.lane.getLastStepVehicleNumber(dut_lane) <= 1:
            return None

        # Vehicle in front of DUT
        dut_pos = traci.vehicle.getLanePosition(constants.DUT)
        data = []
        for vid in traci.lane.getLastStepVehicleIDs(dut_lane):
            if vid == constants.DUT:
                continue
            pos = traci.vehicle.getLanePosition(vid)
            s = pd.Series({
                "vid" : vid,
                "pos" : pos
            })
            data.append(s)

        df = pd.DataFrame(data)
        df = df[df["pos"] > dut_pos].sort_values(by="pos",ascending=True)
        if len(df.index) == 0:
            return None
        return df.iloc[0]["vid"]

    def check_for_new_stops(self):
        cur = traci.vehicle.getSpeed(constants.DUT)
        if len(self.dut_speed_history) > 0:
            prev = self.dut_speed_history[-1]
            if prev != 0 and cur == 0:
                self.score["n stops"] += 1
        self.dut_speed_history.append(cur)
        return
    
    def collision_metrics(self):
        collisions = traci.simulation.getCollisions()
        for c in collisions:
            c : traci._simulation.Collision
            if constants.DUT in [c.collider, c.victim]:
                self.score["collisions"].append( self.collision2dict(c) )
            continue
        return

    def collision2dict(self, c : traci._simulation.Collision) -> dict:
        assert constants.DUT in [c.collider, c.victim]
        
        # print(c)
        # print() 

        data = {
            "time" : self.get_time(),
            "pos" : c.pos,
            "lane" : c.lane
        }
        if constants.DUT == c.collider:
            data["status"] = "collider"
            data["speed"] = c.colliderSpeed
            data["other type"] = c.victimType
            data["other speed"] = c.victimSpeed
        else:
            data["status"] = "victim"
            data["speed"] = c.victimSpeed
            data["other type"] = c.colliderType
            data["other speed"] = c.colliderSpeed

        # print(data)
        # print()

        # quit()
        return data
    
    def get_foes_in_intersection(self) -> list[str]:
        foes = [vid for vid in traci.vehicle.getIDList() if not \
            ((vid == constants.DUT) \
             or (traci.vehicle.getLaneID(vid)[0] != ":"))]
        return foes

    def get_time(self) -> float:
        """
        Get simualtion time, adjust for initializiton
        """
        return traci.simulation.getTime() - self.start_time

    def add_passenger_polygons(self):

        for vid in traci.vehicle.getIDList():
            center = traci.vehicle.getPosition(vid)            
            rotation = traci.vehicle.getAngle(vid)

            # Adjust for SUMO axes
            if vid[0] in "ns":
                rotation += 90
            else:
                rotation -= 90

            # Construct a polygon
            polygon = utils.passenger_polygon(rotation, center)

            # Choose the color
            if constants.sumo.show_polygons:
                if constants.sumo.override_polygon_color:
                    color = constants.sumo.polygon_color
                else:
                    if vid == constants.DUT:
                        color = constants.RGBA.light_blue
                    elif traci.vehicle.getTypeID(vid) == "AggrCar":
                        color = constants.RGBA.red
                    else:
                        color = constants.RGBA.yellow
            else:
                color = constants.RGBA.black

            # Add Polygon
            pid = vid
            traci.polygon.add(
                pid, 
                list(polygon.exterior.coords),
                color,
                layer=5,
                lineWidth=0.1
            )

            # Attach to a vehicle
            traci.polygon.addDynamics(
                pid,
                vid,
                rotate = True
            )
            continue
        return

    def clear_polygons(self):
        for pid in traci.polygon.getIDList():
            traci.polygon.remove(pid)
        return

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
            typeID= constants.traci.gamma_cross.dut_type,
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
        pos = turn_lane_length - 20 - 2*7 - 20
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
            if constants.sumo.track_dut:
                traci.gui.trackVehicle(constants.sumo.default_view, constants.DUT)
            traci.gui.setZoom(
                constants.sumo.default_view, 
                constants.sumo.dut_zoom
            )
        # input("hhh")
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
        for i in range(1,4+1):
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

        pos_offset = {
            1 : 20,
            2 : 20 + 7,
            3 : 20 + 2*7 + 25 + 22,
            4 : 20 + 2*7 + 25 + 22 + 7
        }

        # Move to the correct edges
        for i in range(len(df.index)):
            s = df.iloc[i]
            
            ipos = int(s["vid"][-1])
            lane_length = constants.traci.gamma_cross.turn_lane_length
            pos = lane_length - pos_offset[ipos]
            # print(pos)
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
