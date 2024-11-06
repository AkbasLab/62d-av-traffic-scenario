seed = 12312
n_tests = 10_000



class RGBA:
    light_blue = (12,158,236,255)
    rosey_red = (244,52,84,255)
    cyan = (0,255,255,255)
    lime = (0,255,0,255)
    black = (0,0,0,255)
    red = (255,0,0,255)
    yellow = (255,255,0,255)
class sumo:
    gui = False
    start = True
    quit_on_end = True
    pause_after_initialze = False
    track_dut = False
    delay_ms = 50
    action_step_length = 0.1
    step_length = 0.1
    quiet_mode = True
    dut_zoom = 800
    lane_change_duration = 0.5
    show_polygons = True
    override_polygon_color = False
    polygon_color = RGBA.lime
    error_log_file = "log/error.txt"
    gui_setting_file = "sumo_config/gui.xml"
    init_state_file = "temp/init-state.xml"
    default_view = 'View #0'


class vehicle_types:
    aggresive = "AggrCar"
    conservative = "Car"

class traci:
    default_lane_change_behavior = 1621
    class gamma_cross:
        dut_route = "eb_right"    
        turn_lane_length = 200
        dut_type = vehicle_types.aggresive
        net_file = "sumo_config/gamma_cross/cross3l.net.xml"
        route_files = "sumo_config/gamma_cross/cross3l.rou.xml"
        config = {
            "--net-file" : net_file,
            "--route-files" : route_files,
        }
        tl_order = {
            "sb_right" : 0,
            "sb_straight" : 1,
            "sb_left" : 2,
            "wb_right" : 3,
            "wb_straight" : 4,
            "wb_left" : 5,
            "nb_right" : 6,
            "nb_straight" : 7,
            "nb_left" : 8,
            "eb_right" : 9,
            "eb_straight" : 10,
            "eb_left" : 11
        }
        internal_lanes = {
            "eb_left" : ":0_11_0",
            "eb_straight" : ":0_10_0",
            "eb_right" : ":0_9_0"
        }
        
        


DUT = "dut"
FOE = "foe"