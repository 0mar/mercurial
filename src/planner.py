__author__ = 'omar'

import networkx as nx
import numpy as np

import functions as ft
from geometry import LineSegment, Path, Point, Coordinate, Interval, Velocity
from pedestrian import Pedestrian
from scene import Obstacle


class Planner:
    """
    Naive planner module. Implements a home-made path planning algorithm (with poor performances).
    Main use is to be extended by better modules, like the GraphPlanner below.
    Some methods are still usable, like get_goal and collective_update.

    Lower level modules should be agnostic to the path planning module.
    Therefore, Pedestrian and Scene class members are generated and modified on the fly.
    """
    on_track = 0
    reached_checkpoint = 1
    other_state = 2

    def __init__(self, scene):
        """
        Creates a global path planner for the scene.
        :param scene: Scene filled with obstacles, goals and pedestrians

        :return: Path planner that has planned out paths for each pedestrian.
        """
        self.scene = scene
        for pedestrian in self.scene.pedestrian_list:
            pedestrian.path = self.create_path(pedestrian, self.scene.exit_obs)
            pedestrian.line = pedestrian.path.pop_next_segment()
            # pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position.array)

    def create_path(self, pedestrian: Pedestrian, goal_obstacle) -> Path:
        """
        Deprecated method for creating a path for each pedestrian from his current position
        to his goal obstacle.
        :param pedestrian: Pedestrian located in scene
        :param goal_obstacle: Goal (in scene) reachable from the pedestrians location.
        :return: Path (sequence of Line Segments) leading from the pedestrian to the goal
        """
        path_to_exit = Path([])
        sub_start = pedestrian.position
        while not (sub_start in goal_obstacle):
            goal = Planner.get_closest_goal_position(sub_start, goal_obstacle)
            line_to_goal = LineSegment([sub_start, goal])
            angle_to_goal = (goal - sub_start).angle
            colliding_obstacles = []
            for obstacle in self.scene.obstacle_list:
                if obstacle.in_interior and line_to_goal.crosses_obstacle(obstacle):
                    colliding_obstacles.append(obstacle)
            if colliding_obstacles:
                sub_finish = self.get_intermediate_goal(sub_start, angle_to_goal, colliding_obstacles)
            else:
                sub_finish = goal
            path_to_exit.append(LineSegment([sub_start, sub_finish]))
            sub_start = sub_finish
        return path_to_exit

    @staticmethod
    def get_closest_goal_position(position, goal_obstacle):
        """
        Computes the position of the goal obstacle closest to the provided position
        If the position provided lies in the goal, returns pedestrian coordinates.
        Otherwise, returns a Point in goal such that distance from position to Point is minimal
        :param position: Point somewhere in scene
        :param goal_obstacle: some (goal) obstacle
        :return: Point somewhere in obstacle
        """
        # Checking whether the x or y component already lies between the bounds of object.
        # Otherwise, desired component is the closest corner.
        goal_dim = [0, 0]
        for dim in range(2):
            obs_interval = Interval([goal_obstacle.begin[dim], goal_obstacle.end[dim]])
            pos_dim = position[dim]
            if pos_dim in obs_interval:
                goal_dim[dim] = position[dim]
            elif pos_dim < obs_interval.begin:
                goal_dim[dim] = obs_interval.begin
            else:
                goal_dim[dim] = obs_interval.end
        return Point(goal_dim)

    @staticmethod

    def get_intermediate_goal(start, angle_to_goal, obstacles, safe_distance=3.):
        """
        Deprecated method, used in old path planning method.
        :param start:
        :param angle_to_goal:
        :param obstacles:
        :param safe_distance:
        :return:
        """
        raise DeprecationWarning("Using this method leads to poor path planning results.")
        # In this method, 0 === False === Left, 1 === True === Right
        corner_points = [None, None]  # max corners left and right from destination
        max_angles = [0, 0]
        for obstacle in obstacles:
            for corner, corner_location in obstacle.corner_info_list:
                angle_to_corner = (corner - start).angle - angle_to_goal
                if angle_to_corner < - np.pi:  # Corner not in front of us
                    angle_to_corner += 2 * np.pi
                direction = int(angle_to_corner < 0)
                if np.abs(max_angles[direction]) <= np.abs(angle_to_corner):
                    max_angles[direction] = angle_to_corner
                    corner_points[direction] = (corner, corner_location)
        best_direction = int(max_angles[0] >= -max_angles[1])
        best_corner = corner_points[best_direction][0]
        obstacle_repulsion = np.sign(np.array(corner_points[best_direction][1]) - 0.5)
        return best_corner + Point(obstacle_repulsion) * safe_distance

    @staticmethod
    def get_path_length(pedestrian):
        """
        Compute the sum of Euclidian lengths of the path line segments
        :param pedestrian: owner of the path
        :return: Length of planned path
        """
        if not pedestrian.line:
            return 0
        length = pedestrian.line.length
        for line in pedestrian.path:
            length += line.length
        return length

    def collective_update(self):
        """
        The update step of the simulation
        # 1: Progresses one time step.
        # 2: Moves the pedestrians according to desired velocity, group velocity and UIC
                * Account for MDE
        # 3: Corrects for obstacles and walls
        # 3.5: Check all pedestrians if arrived at location
        # 3.75: Check if path is still correct (using last location and such)
        # 4: Provides the new desired velocity.
        # 5: Enforce minimal distance between pedestrians.
        :return: None
        """
        # 1
        self.scene.time += self.scene.dt
        # 2
        self.scene.move_pedestrians()

        # 3
        for pedestrian in self.scene.pedestrian_list:
            if self.scene.alive_array[pedestrian.counter]:
                pedestrian.correct_for_geometry()

        stationary_pedestrian_array = self.scene.get_stationary_pedestrians()
        for pedestrian in self.scene.pedestrian_list:
            if not stationary_pedestrian_array[pedestrian.counter] and self.scene.alive_array[pedestrian.counter]:
                # assert self.scene.is_accessible(pedestrian.position)
                pedestrian.move_to_position(Point(pedestrian.line.end), self.scene.dt)
                remaining_path = pedestrian.line.end - pedestrian.position
                allowed_range = 0.5  # some experimental threshold based on safety margin of obstacles
                checkpoint_reached = np.linalg.norm(remaining_path) < allowed_range
                done = pedestrian.is_done()
                if done:
                    self.scene.remove_pedestrian(pedestrian)
                    continue
                if checkpoint_reached:  # but not done
                    if pedestrian.path:
                        pedestrian.line = pedestrian.path.pop_next_segment()
                        pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position)
                else:
                    pedestrian.velocity = Velocity(remaining_path)  # Expensive...
            elif self.scene.alive_array[pedestrian.counter]:
                # Stationary pedestrian. Creating new path.
                pedestrian.path = self.create_path(pedestrian, self.scene.exit_obs)
                pedestrian.line = pedestrian.path.pop_next_segment()
                pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position.array)


class GraphPlanner(Planner):
    """
    Upgraded path planner, based on A* algorithm.
    Converts the scene to a graph, adds the pedestrians location and finds the shortest path to the goal.
    In this implementation, only one exit is present.
    This should not be hard to generalize, just keep an eye on the runtime.
    """
    def __init__(self, scene):
        """
        Constructs a Graph planner
        :param scene: Scene filled with obstacles, goals and pedestrians
        :return: Graph planner object with planned paths for each pedestrian
        Note that the path is 'popped' in the constructor.
        """
        self.scene = scene
        self.graph = None
        self._create_obstacle_graph(self.scene.exit_obs)
        ft.log("Started preprocessing global paths")
        for pedestrian in scene.pedestrian_list:
            pedestrian.path = self.create_path(pedestrian, self.scene.exit_obs)
            pedestrian.line = pedestrian.path.pop_next_segment()
            pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position.array)
        ft.log("Finished preprocessing global paths")

    def create_path(self, pedestrian: Pedestrian, goal_obstacle) -> Path:
        """
        Creates a path leading from pedestrian location to finish
        :param pedestrian: Pedestrian under consideration
        :param goal_obstacle: Goal of the pedestrian
        :return: Path if goal_obstacle is reachable from pedestrian location
        :raise: RunTimeError when no path can be computed.
        """
        ped_graph = nx.Graph(self.graph)
        ped_graph.add_node(pedestrian.position)
        self._fill_with_required_edges(pedestrian.position, ped_graph, goal_obstacle)
        try:
            path = nx.astar_path(ped_graph, pedestrian.position, goal_obstacle)
        except nx.NetworkXNoPath:
            raise RuntimeError("No path could be found from %s to exit. Check your obstacles" % pedestrian)
        path_to_exit = Path([])
        prev_point = path[0]
        for point in path[1:-1]:
            line = LineSegment([prev_point, point])
            path_to_exit.append(line)
            prev_point = point
        finish_point = Planner.get_closest_goal_position(prev_point, goal_obstacle)
        line_to_finish = LineSegment([prev_point, finish_point])
        # assert self.line_crosses_no_obstacles(line_to_finish)
        path_to_exit.append(line_to_finish)
        # print ("Path: %s\nPath obj %s"%(path,path_to_exit))
        return path_to_exit

    def _create_obstacle_graph(self, goal_obstacle):
        """
        Create the graph of the obstacles. Details on implementation are found in the report.
        :param goal_obstacle: The goal under consideration.
        :return:
        """
        self.graph = nx.Graph()
        self.graph.add_node(goal_obstacle)
        for obstacle in self.scene.obstacle_list:
            if obstacle.in_interior and not obstacle.permeable:
                ordered_list = [corner + margin for corner, margin in zip(obstacle.corner_list, obstacle.margin_list)]
                for point in ordered_list:
                    # if all(point.array < self.scene.size.array): # Hypocrite check.
                    self.graph.add_node(point)
        for node in self.graph.nodes():
            if node is not goal_obstacle:
                self._fill_with_required_edges(node, self.graph, goal_obstacle)

    def _fill_with_required_edges(self, node: Point, graph, goal_obstacle):
        """
        Fills the graph (which now only consists of nodes) with the edges connecting to the given node
        :param node: Obstacle node under consideration
        :param graph: Graph the node is part of
        :param goal_obstacle: Goal in the scene
        :return:
        """
        goal_point = Planner.get_closest_goal_position(node, goal_obstacle)
        line_to_goal = LineSegment([node, goal_point])
        path_free = True
        for obstacle in self.scene.obstacle_list:
            if obstacle.in_interior and line_to_goal.crosses_obstacle(obstacle, open_sets=True):
                path_free = False
                break
        if path_free:
            graph.add_edge(u=node, v=goal_obstacle, weight=line_to_goal.length)
        for other_node in graph.nodes():
            if (other_node is not node) and (other_node is not goal_obstacle) and self.scene.is_within_boundaries(
                    other_node):
                path = LineSegment([node, other_node])
                path_free = True
                for obstacle in self.scene.obstacle_list:
                    if obstacle.in_interior and path.crosses_obstacle(obstacle, open_sets=True):
                        path_free = False
                        break
                if path_free:
                    graph.add_edge(u=node, v=other_node, weight=path.length)

    def draw_graph(self, graph, pedestrian=None):
        """
        Draws a graph, labeling the scene exit and a potential pedestrian.
        :param graph: a graph (duh)
        :param pedestrian: Pedestrians location in the scene
        :return:
        """
        pos = {}
        labeling = {}
        node_colors = []
        draw_graph = nx.Graph(graph)
        if pedestrian:
            draw_graph.add_node(pedestrian.position)
            labeling[pedestrian.position] = r'$a$'
            self._fill_with_required_edges(pedestrian.position, draw_graph, self.scene.exit_obs)
        for node in draw_graph.nodes():
            position = None
            if pedestrian and node == pedestrian.position:
                node_colors.append('gray')
                position = node.array
            elif isinstance(node, Coordinate):
                node_colors.append('black')
                position = node.array
            elif isinstance(node, Obstacle):
                position = (node.begin + node.end) * 0.5
                node_colors.append('red')
            else:
                raise ValueError("Unable to plot %s node type" % type(node))
            pos[node] = position
        labeling[self.scene.exit_obs] = r'$E$'
        nx.draw_networkx_nodes(draw_graph, pos, node_color=node_colors, label=labeling, node_size=500)
        nx.draw_networkx_edges(draw_graph, pos)
        nx.draw_networkx_labels(draw_graph, pos, labeling, font_size=16)
