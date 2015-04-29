__author__ = 'omar'

import networkx as nx
import matplotlib.pyplot as plt
from geometry import *
class GT:

    def __init__(self,scene):
        self.scene = scene
        self.graph = None
        self.goal = scene.exit_obs
        self._create_obstacle_graph()
        self.create_path(scene.ped_list[0])

    def create_path(self,pedestrian):
        ped_graph = self.graph.copy()
        print("Same graph %s same objects \n\n%s\n\n%s"%(ped_graph==self.graph,ped_graph.nodes(),self.graph.nodes()))
        ped_graph.add_node(pedestrian.position)
        self.fill_with_edges(pedestrian.position,ped_graph)
        path_to_exit = nx.astar_path(ped_graph,pedestrian.position,self.goal)
        print(path_to_exit)

    def _create_obstacle_graph(self):
        self.graph = nx.Graph()
        self.graph.add_node(self.goal)
        for obstacle in self.scene.obs_list:
            if obstacle.in_interior:
                ordered_list = [corner[0] for corner in obstacle.corner_info_list]
                ordered_list[2], ordered_list[3] = ordered_list[3], ordered_list[2]
                for index in range(len(ordered_list)):
                    self.graph.add_edge(u=ordered_list[index-1],v=ordered_list[index])
                    # Also initializes edges; prettier
        for node in self.graph.nodes():
            self.fill_with_edges(node,self.graph)

    def fill_with_edges(self,node,graph):
        if node is not self.goal:
            if isinstance(node,Exit):
                raise AssertionError("WHAAAAAH")
            goal_point = Planner.get_goal(node,self.goal)
            line_to_goal = LineSegment([node,goal_point])
            path_free = True
            for obstacle in self.scene.obs_list:
                if obstacle.in_interior and line_to_goal.crosses_obstacle(obstacle,strict=True):
                    path_free = False
                    break
            if path_free:
                graph.add_edge(u=node,v=self.goal,weight=line_to_goal.length)
            for other_node in graph.nodes():
                if (other_node is not node) and (other_node is not self.goal):
                    if isinstance(other_node,Exit):
                        print("%s"%(other_node == self.goal))
                        raise AssertionError("WHAAAAAH")
                    path = LineSegment([node,other_node])
                    for obstacle in self.scene.obs_list:
                        path_free=True
                        if obstacle.in_interior and path.crosses_obstacle(obstacle,strict=True):
                            path_free=False
                            break
                    if path_free:
                        graph.add_edge(u=node,v=other_node,weight=path.length)

    def draw_graph(self):
        pos = {}
        for node in self.graph.nodes():
            position = None
            if isinstance(node,Point):
                position = node.array
            elif isinstance(node,Obstacle):
                position = (node.begin + node.end)*0.5
            else:
                raise ValueError("Wrong node types in the graph")
            pos[node] = position
        nx.draw_networkx_nodes(self.graph,pos)
        nx.draw_networkx_edges(self.graph,pos)

scene = Scene(size=Size([250, 150]), pedNumber=1)
gt = GT(scene)
gt.create_obstacle_graph()
