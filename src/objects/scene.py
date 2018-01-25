import numpy as np
from lib.wdt import map_image_to_costs, get_weighted_distance_transform
from math_objects.geometry import Point, Size
from scipy.ndimage import zoom


class Scene:
    """
    Models a scene. A scene is a rectangular object with obstacles and pedestrians inside.
    """

    def __init__(self, params):
        """
        Initializes a Scene using the settings in the configuration file augmented with command line parameters
        :return: scene instance.
        """
        self.time = 0
        self.counter = 0
        self.total_pedestrians = 0
        self.size = None
        self.params = params

        self.on_step_functions = []
        self.on_pedestrian_exit_functions = []
        self.on_pedestrian_init_functions = []

        self.fire = None
        # self.gutter_cells = self.get_obstacle_gutter_cells()
        # Array initialization
        self.position_array = self.last_position_array = self.velocity_array = None
        self.acceleration_array = self.max_speed_array = self.active_entries = None
        self.pedestrian_list = []
        self.index_map = {}
        self.env_field = self.direction_field = None
        self.dx = self.dy = None

    def prepare(self):
        """
        Method called directly before simulation start. All parameters need to be registered.
        :return: None
        """
        self.size = Size([self.params.scene_size_x, self.params.scene_size_y])
        self.env_field = np.rot90(map_image_to_costs(self.params.scene_file), -1)
        self.direction_field = get_weighted_distance_transform(self.env_field)
        self.dx = self.size[0] / self.env_field.shape[0]
        self.dy = self.size[1] / self.env_field.shape[1]
        self.position_array = np.zeros([self.total_pedestrians, 2])
        self.last_position_array = np.zeros([self.total_pedestrians, 2])
        self.velocity_array = np.zeros([self.total_pedestrians, 2])
        self.acceleration_array = np.zeros([self.total_pedestrians, 2])
        self.max_speed_array = np.empty(self.total_pedestrians)
        self.active_entries = np.ones(self.total_pedestrians, dtype=bool)

        if self.params.max_speed_distribution.lower() == 'uniform':
            # in a uniform distribution [a,b], sd = (b-a)/sqrt(12).
            interval_size = self.params.max_speed_sd * np.sqrt(12)
            interval_start = interval_size / 2 + self.params.max_speed_av
            self.max_speed_array = interval_start + np.random.rand(self.total_pedestrians) * interval_size
        elif self.params.max_speed_distribution.lower() == 'normal':
            self.max_speed_array = self.params.max_speed_sd * np.abs(
                np.random.randn(self.total_pedestrians)) + self.params.max_speed_av
        else:
            raise NotImplementedError('Distribution %s not yet implemented' % self.params.max_speed_distribution)

    def _expand_arrays(self):
        """
        Doubles the size (first dimension) of a numpy array with the given factor.
        This method is only required in case the number of pedestrians is growing.
        Missing entries are set to zero
        :return: None
        """
        # I don't like [gs]etattr, but this is pretty explicit
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
        self.velocity_array += self.acceleration_array * self.params.dt
        self.position_array += self.velocity_array * self.params.dt

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
            return 0 < self.direction_field[cell] < np.inf
        else:
            return self.direction_field[cell] < np.inf

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
        Removes a pedestrian from the scene and performs cleanup.
        :param pedestrian: The pedestrian instance to be removed.

        :return: None
        """
        index = pedestrian.index
        self.index_map[index] = None
        self.pedestrian_list.remove(pedestrian)
        self.active_entries[index] = False
        for func in self.on_pedestrian_exit_functions:
            func(pedestrian)

    def add_pedestrian(self, pedestrian):
        """
        Adds a new pedestrian to the scene
        :param pedestrian: The pedestrian to be added

        :return: None
        """
        index = pedestrian.index
        self.index_map[index] = pedestrian
        self.pedestrian_list.append(pedestrian)
        self.active_entries[index] = True
        for func in self.on_pedestrian_init_functions:
            func(pedestrian)

    def get_obstacles(self, nx, ny):
        """
        Determines the obstacles for a (smaller) resolution of the image.
        Useful for macroscopic operations like computing pressure and smoke

        :param nx: number of points along x direction
        :param ny: number of points along y direction
        :return: resized numpy array with shape (nx,ny)
        """
        obstacle_field = (self.env_field == np.inf).astype(int)
        resized_obstacle_field = zoom(obstacle_field, (nx / obstacle_field.shape[0], ny / obstacle_field.shape[1]))
        return resized_obstacle_field
