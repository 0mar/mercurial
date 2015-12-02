__author__ = 'omar'
import sys

sys.path.insert(1, '../src')

from simulation_manager import SimulationManager
from cython_modules.grid_computer import solve_LCP_with_pgs
from scene import Scene
import numpy as np

demo_file_name = '../scenes/demo_obstacle_list.json'
empty_file_name = '../scenes/empty_scene.json'

from grid_computer import GridComputer


class TestGridComputer:
    def __init__(self):
        self.config = SimulationManager.get_default_config()
        self.config['general']['obstacle_file'] = demo_file_name
        self.scene = Scene(1000, self.config)
        self.grid_computer = GridComputer(self.scene, False, False, False, self.config)

    def test_quad_solver_1(self):
        n = 5
        e = np.ones(n)
        A = np.diag(-e[:-1], 1) + np.diag(-e[:-1], -1) + np.diag(2. * e, 0)
        b = np.array([1, -3, 0, 4, -2])
        real_z = np.array([0.75, 2.5, 1.25, 0, 1])
        real_w = np.array([0, 0, 0, 1.75, 0])
        z = GridComputer.solve_LCP_with_quad(A, b)
        print(z - real_z[:, None])
        w = np.dot(A, z) + b[:, None]
        assert np.dot(w.T, z) < 0.00001
        assert np.allclose(real_z[:, None], z)
        # Maybe set our own tolerance?
        assert np.allclose(real_w[:, None], w)

    def test_quad_solver_2(self):
        n = 7
        A = np.diag(np.arange(7) + 1).astype(float)
        print(A)
        b = np.array([-1, 4, 4, 2, -3, 7, -5]).astype(float) # Seems less stable with zeros in b
        real_z = np.array([1, 0, 0, 0, 0.6, 0, 5 / 7])
        real_w = np.array([0, 4, 4, 2, 0, 7, 0])
        z = GridComputer.solve_LCP_with_quad(A, b)
        print(z)
        print(z - real_z[:, None])
        w = np.dot(A, z) + b[:, None]
        assert np.dot(w.T, z) < 0.00001
        assert np.allclose(real_z[:, None], z)
        assert np.allclose(real_w[:, None], w)


    def test_pgs_solver_1(self):
        n = 5
        e = np.ones(n)
        A = np.diag(-e[:-1], 1) + np.diag(-e[:-1], -1) + np.diag(2. * e, 0)
        b = np.array([1, -3, 0, 4, -2]).astype(float)
        real_z = np.array([0.75, 2.5, 1.25, 0, 1])
        real_w = np.array([0, 0, 0, 1.75, 0])
        z = solve_LCP_with_pgs(A, b)
        print(z - real_z[:, None])
        w = np.dot(A, z) + b[:, None]
        assert np.dot(w.T, z) < 0.00001
        assert np.allclose(real_z[:, None], z)
        # Maybe set our own tolerance?
        assert np.allclose(real_w[:, None], w)

    def test_pgs_solver_2(self):
        n = 7
        A = np.diag(np.arange(7) + 1).astype(float)
        print(A)
        b = np.array([-1, 4, 4, 2, -3, 7, -5]).astype(float) # Seems less stable with zeros in b
        real_z = np.array([1, 0, 0, 0, 0.6, 0, 5 / 7])
        real_w = np.array([0, 4, 4, 2, 0, 7, 0])
        z = solve_LCP_with_pgs(A, b)
        print(z)
        print(z - real_z[:, None])
        w = np.dot(A, z) + b[:, None]
        assert np.dot(w.T, z) < 0.00001
        assert np.allclose(real_z[:, None], z)
        assert np.allclose(real_w[:, None], w)
