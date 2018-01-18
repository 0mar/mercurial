import numpy as np
from lib.wdt import map_image_to_costs, get_weighted_distance_transform
from math_objects.geometry import Point, Size
import params


class Scene:
    """
    Models a scene. A scene is a rectangular object with obstacles and pedestrians inside.
    """

    def __init__(self):
        """
        Initializes a Scene using the settings in the configuration file augmented with command line parameters
        :return: scene instance.
        """
        self.time = 0
        self.counter = 0
        self.total_pedestrians = 0
        self.size = None

        self.on_step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_pedestrian_init_functions = []

        self.fire = None
        # self.fire_center = None # Todo: Implement fires
        # if self.fire_center:
        #     self.fire = Fire(self.size *Point(self.fire_center), self.size[0] * self.fire_radius, self.fire_intensity)
        #     self.drawables.append(self.fire)
        # self.gutter_cells = self.get_obstacle_gutter_cells()
        # Array initialization
        self.position_array = self.last_position_array = self.velocity_array = None
        self.acceleration_array = self.max_speed_array = self.active_entries = None
        self.pedestrian_list = []
        self.index_map = {}

    def prepare(self):
        """
        Method called directly before simulation start. All parameters need to be registered.
        :return: None
        """
        self.size = Size([params.scene_size_x, params.scene_size_y])
        self.env_field = np.rot90(map_image_to_costs(params.scene_file), -1)
        self.direction_field = get_weighted_distance_transform(self.env_field)
        self.dx = self.size[0] / self.env_field.shape[0]
        self.dy = self.size[1] / self.env_field.shape[1]
        self.position_array = np.zeros([self.total_pedestrians, 2])
        self.last_position_array = np.zeros([self.total_pedestrians, 2])
        self.velocity_array = np.zeros([self.total_pedestrians, 2])
        self.acceleration_array = np.zeros([self.total_pedestrians, 2])
        self.max_speed_array = np.empty(self.total_pedestrians)
        self.active_entries = np.ones(self.total_pedestrians, dtype=bool)

        if params.max_speed_distribution.lower() == 'uniform':
            # in a uniform distribution [a,b], sd = (b-a)/sqrt(12).
            interval_size = params.max_speed_sd * np.sqrt(12)
            interval_start = interval_size / 2 + params.max_speed_av
            self.max_speed_array = interval_start + np.random.rand(self.total_pedestrians) * interval_size
        elif params.max_speed_distribution.lower() == 'normal':
            self.max_speed_array = np.abs(np.random.randn(self.total_pedestrians))
        else:
            raise NotImplementedError('Distribution %s not yet implemented' % params.max_speed_distribution)

    # def get_obstacle_gutter_cells(self, radius=2): # Move
    #     """
    #     Compute all the cells which lie of distance `radius` from an obstacle.
    #     Used in the macroscopic pressure, in case we want to repel/attract pedestrians from/to specific zones.
    #     :param radius: distance in cells to the obstacles
    #     :return: array of nx,ny with 1 on gutter cells and obstacle cells and 0 on rest.
    #     """
    #     if not self.snap_obstacles:
    #         ft.warn("Computing obstacle coverage: Snapping is turned off, so this is only an estimation")
    #     dx, dy = self.config['general'].getfloat('cell_size_x'), self.config['general'].getfloat('cell_size_y')
    #     nx = int(self.config['general'].getint('scene_size_x') / dx)
    #     ny = int(self.config['general'].getint('scene_size_y') / dy)
    #     obstacle_gutter = np.zeros(self.obstacle_coverage.shape)
    #
    #     for row, col in np.ndindex((nx, ny)):
    #         if not self.obstacle_coverage[row, col]:
    #             right, up = min(row + radius + 1, nx - 1), min(col + radius + 1, ny - 1)
    #             left, down = max(row - radius, 0), max(col - radius, 0)
    #             if np.any(self.obstacle_coverage[left:right, down:up]):
    #                 obstacle_gutter[row, col] = 1
    #     return obstacle_gutter


    def _expand_arrays(self):
        """
        Doubles the size (first dimension) of a numpy array with the given factor.
        This method is only required in case the number of pedestrians is growing.
        Missing entries are set to zero
        :return: None
        """
        # I don't like [gs]etattr, but this is pretty explicit
        # Todo: Is it possible to to make one method that expands all arrays application-wide?
        attr_list = ["position_array", "last_position_array", "velocity_array",
                     "max_speed_array", "active_entries", "aware_pedestrians",
                     "acceleration_array"]
        for attr in attr_list:
            array = getattr(self, attr)
            addition = np.zeros(array.shape, dtype=array.dtype)
            setattr(self, attr, np.concatenate((array, addition), axis=0))
        self.index_map.update({len(self.index_map) + i: None for i in range(len(self.index_map))})

    def move(self):
        """
        Performs a vectorized move of all the pedestrians.
        Assumes that all accelerations and velocities have been set accordingly.
        :return: None
        """
        self.last_position_array = np.array(self.position_array)
        self.velocity_array += self.acceleration_array * params.dt
        self.position_array += self.velocity_array * params.dt

    def correct_for_geometry(self):
        """
        Performs a vectorized correction to make sure pedestrians do not enter obstacles
        :return: None
        """
        ge_zero = np.logical_and(self.position_array[:, 0] > 0, self.position_array[:, 1] > 0)
        le_size = np.logical_and(self.position_array[:, 0] < self.size[0], self.position_array[:, 1] < self.size[1])
        still_correct = np.logical_and(ge_zero, le_size)
        cells = (self.position_array // (self.dx, self.dy)).astype(int) % self.env_field.shape
        # All peds for which the modulo kicks in, are already out
        not_in_obstacle = self.env_field[cells[:, 0], cells[:, 1]] < np.inf
        still_correct = np.logical_and(still_correct, not_in_obstacle)
        still_correct = np.logical_and(still_correct, self.active_entries)
        self.position_array += np.logical_not(still_correct)[:, None] * (self.last_position_array - self.position_array)

    def get_stationary_pedestrians(self):
        """
        Computes which pedestrians have not moved since the last time step
        :return: nx1 boolean np.array, True if (existing) pedestrian is stationary, False otherwise
        """
        pos_difference = np.linalg.norm(self.position_array - self.last_position_array, axis=1)
        not_moved = np.logical_and(pos_difference == 0, self.active_entries)
        return not_moved

    def is_within_boundaries(self, coord: Point):
        """
        Check whether a single point lies within the scene.
        :param coord: Point under consideration
        :return: True if within scene, false otherwise
        """
        within_boundaries = all(np.array([0, 0]) < coord.array) and all(coord.array < self.size.array)
        return within_boundaries

    def is_accessible(self, coord: Point, at_start=False):
        """
        Checking whether the coordinate present is an accessible coordinate on the scene.
        When evaluated at the start, the exit is not an accessible object,
        to prevent pedestrians from being spawned there.
        :param coord: Coordinates to be checked
        :param at_start: Whether to be evaluated at the start
        :return: True if accessible, False otherwise.
        """
        if not self.is_within_boundaries(coord):
            return False
        cell = (int(coord[0] // self.dx), int(coord[1] // self.dy))
        if at_start:
            return 0 < self.env_field[cell] < np.inf  # Todo: Rather, you want to check the potential field
        else:
            return self.env_field[cell] < np.inf

    def step(self):
        """
        Compute all step functions in scene not related to planner functions.
        :return: None
        """
        [step() for step in self.on_step_functions]

    def find_finished(self):
        """
        Finds all pedestrians that have reached the goal in this time step.
        If there is a cap on the number of pedestrians allowed to exit,
        we randomly sample that number of pedestrians and leave the rest in the exit.
        If any pedestrians are unable to exit, we set the exit to inaccessible.
        The pedestrians that leave are processed and removed from the scene
        :return: None
        """
        cells = (self.position_array // (self.dx, self.dy)).astype(int) % self.env_field.shape
        in_goal = self.env_field[cells[:, 0], cells[:, 1]] == 0
        finished = np.logical_and(in_goal, self.active_entries)
        index_list = np.where(finished)[0]  # possible extension: Goal cap
        for index in index_list:
            finished_pedestrian = self.index_map[index]
            self.remove_pedestrian(finished_pedestrian)

    def remove_pedestrian(self, pedestrian):
        """
        Removes a pedestrian from the scene.
        :param pedestrian: The pedestrian instance to be removed.
        :return: None
        """
        index = pedestrian.index
        self.index_map[index] = None
        self.pedestrian_list.remove(pedestrian)
        self.active_entries[index] = False
        for func in self.on_pedestrian_exit_functions:
            func(pedestrian)
