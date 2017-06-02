#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import evolve

# Parameters:
D = 0.2
v_x,v_y = 0.2,-0.6
nx = ny = 40
dx = dy = 0.1
dt = 0.2

obstacles = np.zeros([nx,ny])

# Fire distribution
c = 2
f = np.array([c*np.exp(-((i*dx-nx*dx/2)**2+(j*dy-ny*dy/2)**2)*c**2) for i in range(nx) for j in range(ny)])

sparse_a = evolve.get_sparse_matrix(D,v_x,v_y,dx,dy,dt,obstacles)

u = np.zeros(nx*ny)
u_old = np.zeros(nx*ny)

evolve.iterate_jacobi(*sparse_a,f,u_old,obstacles)

