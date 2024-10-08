seed = 123
n_tests = 100



class RGBA:
    light_blue = (12,158,236,255)
    rosey_red = (244,52,84,255)
    cyan = (0,255,255,255)
    lime = (0,255,0,255)
    black = (0,0,0,255)
    red = (255,0,0,255)
    yellow = (255,255,0,255)
class sumo:
    gui = True
    start = True
    quit_on_end = True
    delay_ms = 150
    action_step_length = 0.1
    step_length = 0.1
    quiet_mode = True
    dut_zoom = 800
    lane_change_duration = 0.5
    show_polygons = True
    override_polygon_color = True
    polygon_color = RGBA.lime
    error_log_file = "log/error.txt"
    gui_setting_file = "sumo_config/gui.xml"
    init_state_file = "temp/init-state.xml"
    default_view = 'View #0'

class traci:
    default_lane_change_behavior = 1621
    class gamma_cross:
        dut_route = "eb_left"
        turn_lane_length = 200
        net_file = "sumo_config/gamma_cross/cross3l.net.xml"
        route_files = "sumo_config/gamma_cross/cross3l.rou.xml"
        config = {
            "--net-file" : net_file,
            "--route-files" : route_files,
        }


DUT = "dut"
FOE = "foe"