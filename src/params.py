"""
Contains all parameters.
"""

# general
result_dir = 'results/'
scene_file = 'scenes/test.png'
log_file = 'logs'
dt = 0.1
scene_size_x = 100
scene_size_y = 100
max_speed_av = 1.1
max_speed_sd = 0.4
max_speed_distribution = 'normal'
tolerance = 0.0001
pedestrian_size = 0.4
minimal_distance = 0.7
max_time = 0
max_percentage = 1

# Environment
obstacle_clearance = 4

# pressure
pressure_dx = 2
pressure_dy = 2
smoothing_length = 1
packing_factor = 0.8
min_density = 2
max_density = 8
boundary_pressure = 1  # TODO: Get verified parameter value/relation to scene. Factors: discr size, num_ped, min_dist

# visual
time_delay = 1
screen_size_x = 1000
screen_size_y = 800

# Fire
fire_intensity = 2
# smoke
smoke_dx = 2
smoke_dy = 2
diffusion = 0.4
velocity_x = 0.3
velocity_y = 0.2
smoke_limit = 30
min_speed_ratio = 0.1
max_smoke_level = 30

# following
follow_radius = 5
minimal_follow_radius = 0.2
random_force = 1
