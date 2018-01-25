class Parameters:
    """
    Contains all parameters.
    Todo: I thought I could import this module and change its values globally at runtime... not possible.
    This should become a class, maybe static.
    """

    def __init__(self):
        # general
        self.result_dir = 'results/'
        self.scene_file = 'scenes/test.png'
        self.log_file = 'logs'
        self.dt = 0.1
        self.scene_size_x = 100
        self.scene_size_y = 100
        self.max_speed_av = 0.7
        self.max_speed_sd = 0.1
        self.max_speed_distribution = 'normal'
        self.tolerance = 0.0001
        self.pedestrian_size = 0.8
        self.minimal_distance = 1
        self.max_time = 0
        self.max_percentage = 1

        # Environment
        self.obstacle_clearance = 4

        # pressure
        self.pressure_dx = 2
        self.pressure_dy = 2
        self.smoothing_length = 1
        self.packing_factor = 0.8
        self.min_density = 2
        self.max_density = 8
        self.boundary_pressure = 1  # TODO: Get verified parameter value/relation to scene. Factors: discr size, num_ped, min_dist

        # visual
        self.time_delay = 1
        self.screen_size_x = 1000
        self.screen_size_y = 800

        # Fire
        self.fire_intensity = 0.0001
        # smoke
        self.smoke = True
        self.smoke_dx = 2
        self.smoke_dy = 2
        self.diffusion = 0.4
        self.velocity_x = 0.3
        self.velocity_y = 0.2
        self.smoke_limit = 30
        self.min_speed_ratio = 0.1
        self.max_smoke_level = 30

        # following
        self.follow_radius = 5
        self.minimal_follow_radius = 0.2
        self.random_force = 1
