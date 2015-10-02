__author__ = 'omar'

import networkx as nx
import numpy as np

import functions as ft
from geometry import LineSegment, Path, Point, Interval, Velocity, Coordinate
from scene import Obstacle


class GraphPlanner:
    """
    Upgraded path planner, based on A* algorithm.
    Converts the scene to a graph, adds the pedestrians location and finds the shortest path to the goal.
    In this implementation, only one exit is present.
    This should not be hard to generalize, just keep an eye on the runtime.

    Lower level objects should be agnostic of the planner.
    Therefore, this method creates and modifies Scene and Pedestrian attributes on the fly.
    """

    def __init__(self, scene):
        """
        Constructs a Graph planner
        :param scene: Scene filled with obstacles, goals and pedestrians
        :return: Graph planner object with planned paths for each pedestrian
        Note that the path is 'popped' once in the constructor.
        """
        self.scene = scene
        self.graph = None
        self._create_obstacle_graph()
        ft.log("Started preprocessing global paths")
        for pedestrian in scene.pedestrian_list:
            pedestrian.path = self.create_path(pedestrian)
            pedestrian.line = pedestrian.path.pop_next_segment()
            pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position.array)
        ft.log("Finished preprocessing global paths")

    def _create_obstacle_graph(self):
        """
        Create the graph of the obstacles. Details on implementation are found in the report.
        :param goal_obstacle: The goal under consideration.
        :return:
        """
        self.graph = nx.Graph()
        for goal in self.scene.exit_set:
            self.graph.add_node(goal)
        for obstacle in self.scene.obstacle_list:
            if obstacle.in_interior and not obstacle.permeable:
                ordered_list = [corner + margin for corner, margin in zip(obstacle.corner_list, obstacle.margin_list)]
                for point in ordered_list:
                    self.graph.add_node(point)
        for node in self.graph.nodes():
            if node not in self.scene.exit_set:
                self._fill_with_required_edges(node, self.graph)

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

    def _fill_with_required_edges(self, node: Point, graph):
        """
        Fills the graph (which now only consists of nodes) with the edges connecting to the given node
        :param node: Obstacle node under consideration
        :param graph: Graph the node is part of
        :return:
        """
        for goal in self.scene.exit_set:
            goal_point = GraphPlanner.get_closest_goal_position(node, goal)
            line = LineSegment([node, goal_point])
            path_free = True
            for obstacle in self.scene.obstacle_list:
                if obstacle.in_interior and line.crosses_obstacle(obstacle, open_sets=True):
                    path_free = False
                    break
            if path_free:
                graph.add_edge(u=node, v=goal, weight=line.length)
        for other_node in graph.nodes():
            if (other_node is not node) and (other_node not in self.scene.exit_set) and self.scene.is_within_boundaries(
                    other_node):
                path = LineSegment([node, other_node])
                path_free = True
                for obstacle in self.scene.obstacle_list:
                    if obstacle.in_interior and path.crosses_obstacle(obstacle, open_sets=True):
                        path_free = False
                        break
                if path_free:
                    graph.add_edge(u=node, v=other_node, weight=path.length)

    def create_path(self, pedestrian) -> Path:
        """
        Creates a path leading from pedestrian location to finish
        :param pedestrian: Pedestrian under consideration
        :return: Path if goal_obstacle is reachable from pedestrian location
        :raise: RunTimeError when no path can be computed.
        """
        ped_graph = nx.Graph(self.graph)
        ped_graph.add_node(pedestrian.position)
        best_path = None
        shortest_length = np.Inf
        closest_goal = None
        self._fill_with_required_edges(pedestrian.position, ped_graph)
        for goal in self.scene.exit_set:
            try:
                path = nx.astar_path(ped_graph, pedestrian.position, goal)
                length = nx.astar_path_length(ped_graph, pedestrian.position, goal)
                if length < shortest_length:
                    best_path = path
                    closest_goal = goal
                    shortest_length = length
            except nx.NetworkXNoPath:
                continue
        if not best_path:
            raise RuntimeError("No path could be found from %s to exit. Check your obstacles" % pedestrian)
        path_to_exit = Path([])
        prev_point = best_path[0]
        for point in best_path[1:-1]:
            line = LineSegment([prev_point, point])
            path_to_exit.append(line)
            prev_point = point
        finish_point = GraphPlanner.get_closest_goal_position(prev_point, closest_goal)
        line_to_finish = LineSegment([prev_point, finish_point])
        # assert self.line_crosses_no_obstacles(line_to_finish)
        path_to_exit.append(line_to_finish)
        # print ("Path: %s\nPath obj %s"%(path,path_to_exit))
        return path_to_exit

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
                pedestrian.path = self.create_path(pedestrian)
                pedestrian.line = pedestrian.path.pop_next_segment()
                pedestrian.velocity = Velocity(pedestrian.line.end - pedestrian.position.array)

    @staticmethod
    def get_path_length(pedestrian):
        """
        Compute the sum of Euclidean lengths of the path line segments
        :param pedestrian: owner of the path
        :return: Length of planned path
        """
        if not pedestrian.line:
            return 0
        length = pedestrian.line.length
        for line in pedestrian.path:
            length += line.length
        return length

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

            self._fill_with_required_edges(pedestrian.position, draw_graph)
        for node in draw_graph.nodes():
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
        nx.draw_networkx_nodes(draw_graph, pos, node_color=node_colors, label=labeling, node_size=500)
        nx.draw_networkx_edges(draw_graph, pos)
        nx.draw_networkx_labels(draw_graph, pos, labeling, font_size=16)
