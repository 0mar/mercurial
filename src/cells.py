__author__ = 'omar'
import numpy as np

from geometry import Point, Size, LineSegment
import functions as ft
from obstacles import Exit


class Cell:
    """
    Models a cell from the partitioned scene.
    This is to accommodate the UIC computations as well as to perform a more efficient scene evaluation.
    We assume equal sized grid cells.
    This method is largely unused now
    """

    def __init__(self, row: int, column: int, begin: Point, size: Size):
        self.location = (row, column)
        self.begin = begin
        self.size = size
        self.center = self.begin + self.size * 0.5
        self.pedestrian_set = set()
        self.obstacle_set = set()
        self.is_inaccessible = False

    def obtain_relevant_obstacles(self, obstacle_list):
        """
        Obtain a set of all obstacles overlapping with this cell, stored in the class.
        :param obstacle_list: all obstacles in the scene
        :return: None
        """
        # Check if obstacle is contained in cell, or cell contained in obstacle
        for obstacle in obstacle_list:
            if ft.rectangles_intersect(self.begin, self.begin + self.size, obstacle.begin, obstacle.end, True):
                self.obstacle_set.add(obstacle)
                corner_points = [(Point(obstacle.begin + Size([x, y]) * obstacle.size)) for x in range(2) for y in
                                 range(2)]
                corner_points[2], corner_points[3] = corner_points[3], corner_points[2]
                # Create edges
                edge_list = []
                for i in range(4):
                    edge_list.append(LineSegment([corner_points[i], corner_points[i - 1]]))
                    # Check edges for collisions with the obstacle
                edge_list[1] = LineSegment([edge_list[1].end, edge_list[1].begin])
                edge_list[2] = LineSegment([edge_list[2].end, edge_list[2].begin])
                # Reverse the lines, since the rectangles_intersect only accepts ordered rectangles.
                for edge in edge_list:
                    if ft.rectangles_intersect(edge.begin, edge.end, self.begin, self.begin + self.size, True):
                        break
                else:  # If not found, then the cell must be inaccessible.
                    self.is_inaccessible = True

    def get_covered_fraction(self, num_samples=(6, 6)):
        """
        Compute the fraction of the cell covered with obstacles.
        Since an exact computation involves either big shot linear algebra
        or too much case distinctions, we sample the cell space.
        :return: a double approximating the inaccessible space in the cell
        """
        covered_samples = 0
        for i, j in np.ndindex(num_samples):
            for obstacle in self.obstacle_set:
                if not isinstance(obstacle, Exit):
                    if self.begin + Point([i + 0.5, j + 0.5] / np.array(num_samples)) * self.size in obstacle:
                        covered_samples += 1
                        break
        return covered_samples / (num_samples[0] * num_samples[1])

    def add_pedestrian(self, pedestrian):
        """
        Add a pedestrian to the cell
        :param pedestrian, not already in cell
        :return: None
        """
        assert pedestrian not in self.pedestrian_set
        pedestrian.cell = self
        self.pedestrian_set.add(pedestrian)

    def remove_pedestrian(self, pedestrian):
        """
        Remove a pedestrian present in the cell
        :param pedestrian, present in cell
        :return: None
        """
        assert pedestrian in self.pedestrian_set
        self.pedestrian_set.remove(pedestrian)

    def is_accessible(self, coord, at_start=False):
        """
        Computes accessibility of a point, that is whether it is within the boundaries of this cell
        and whether no obstacle is present at that point.
        Use only for coordinates in cell. A warning is provided otherwise.
        :param coord: Coordinate under consideration
        :param at_start: Used for checking accessibilty in exits: at simulation init, this is not allowed.
        :return: True if coordinate is accessible, False otherwise

        """
        if self.is_inaccessible:
            return False
        within_cell_boundaries = all(self.begin.array <= coord.array) and all(
            coord.array <= self.begin.array + self.size.array)
        if not within_cell_boundaries:
            ft.warn('Accessibility of %s requested outside of %s' % (coord, self))
            return False
        if at_start:
            return all([coord not in obstacle for obstacle in self.obstacle_set])
        else:
            return all([coord not in obstacle or obstacle.accessible for obstacle in self.obstacle_set])

    def __contains__(self, coord):
        """
        Check whether a point lies within a cell
        :param coord: Point under consideration
        """
        return all([self.begin[dim] <= coord[dim] <= self.begin[dim] + self.size[dim] for dim in range(2)])

    def __repr__(self):
        """
        :return: Unique string identifier of cell
        """
        return "Cell %s from %s to %s" % (self.location, self.begin, self.begin + self.size)

    def __str__(self):
        """
        :return: String with cell information and contents
        """
        return "Cell %s. Begin: %s. End %s\nPedestrians: %s\nObstacles: %s" % (
            self.location, self.begin, self.begin + self.size, self.pedestrian_set, self.obstacle_set
        )
