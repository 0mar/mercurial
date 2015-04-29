__author__ = 'omar'
from functions import *


class Planner:
    on_track = 0
    reached_checkpoint = 1
    other_state = 2

    def __init__(self, scene):
        fyi("Naive planner implementation")
        self.scene = scene
        for pedestrian in self.scene.ped_list:
            pedestrian.path = self.create_path(pedestrian, self.scene.exit_obs)
            pedestrian.line = pedestrian.path.pop_next_segment()
            pedestrian.state = Planner.on_track

    def create_path(self, pedestrian, goal_obstacle):
        path_to_exit = Path([])
        sub_start = pedestrian.position
        while not (sub_start in goal_obstacle):
            goal = Planner.get_goal(sub_start, goal_obstacle)
            line_to_goal = LineSegment([sub_start, goal])
            angle_to_goal = (goal - sub_start).angle
            colliding_obstacles = []
            for obstacle in self.scene.obs_list:
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
    def get_goal(position, obstacle):
        # Checking whether the x or y component already lies between the bounds of object.
        # Otherwise, desired component is the closest corner.
        goal_dim = [0, 0]
        for dim in range(2):
            obs_interval = Interval([obstacle.begin[dim], obstacle.end[dim]])
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
        # In this method, 0 == False == Left, 1 == True == Right
        corner_points = [None, None]  # max corners left and right from destination
        max_angles = [0, 0]
        for obstacle in obstacles:
            for corner, corner_location in obstacle.corner_info_list:
                angle_to_corner = (corner - start).angle - angle_to_goal
                if angle_to_corner < - np.pi:  # Corner not in front of us
                    angle_to_corner += 2*np.pi
                direction = int(angle_to_corner < 0)
                if np.abs(max_angles[direction]) <= np.abs(angle_to_corner):
                    max_angles[direction] = angle_to_corner
                    corner_points[direction] = (corner, corner_location)
        best_direction = int(max_angles[0] >= -max_angles[1])
        best_corner = corner_points[best_direction][0]
        obstacle_repulsion = np.sign(np.array(corner_points[best_direction][1]) - 0.5)
        return best_corner + Point(obstacle_repulsion) * safe_distance

    def update(self):
        for pedestrian in self.scene.ped_list:
            if pedestrian.state == Planner.on_track:
                checkpoint_reached = pedestrian.move_to_position(Point(pedestrian.line.end), self.scene.dt)
                if checkpoint_reached:
                    pedestrian.state = Planner.reached_checkpoint
            elif pedestrian.state == Planner.reached_checkpoint:
                if pedestrian.path:
                    pedestrian.line = pedestrian.path.pop_next_segment()
                    pedestrian.state = Planner.on_track
                else:
                    pedestrian.state = Planner.other_state
            else:
                if not pedestrian.is_done():
                    pedestrian.path = self.create_path(pedestrian, self.scene.exit_obs)
                else:
                    self.scene.ped_list.remove(pedestrian)
                    fyi("%s has left the building" % pedestrian)

    def plandemo(self):
        for pedestrian in self.scene.ped_list:
            pedestrian.path = self.create_path(pedestrian,self.scene.exit_obs)
