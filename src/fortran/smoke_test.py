#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import evolve
import time

# Parameters:
D = 0.2
v_x,v_y = 0.2,-0.6
nx = ny = 400
dx = dy = 0.1
dt = 0.2
T = 20
t=0
obstacles = np.zeros([nx,ny])
obstacles[1,3]=1

# Fire distribution
c = 2
f = np.array([c*np.exp(-((i*dx-nx*dx/2)**2+(j*dy-ny*dy/2)**2)*c**2) for i in range(nx) for j in range(ny)])

a_v,a_r,a_c,nnz = evolve.get_sparse_matrix(D,v_x,v_y,dx,dy,dt,obstacles)


u = np.zeros(nx*ny)
u_old = np.zeros(nx*ny)
i=0
while t<T:
    i+=1
    t += dt
    u = evolve.iterate_jacobi(a_v,a_r,a_c,nnz,f*dt+u_old,u_old,obstacles)
    plt.imshow(np.rot90(u.reshape(nx,ny)))
    plt.savefig("image-%003d.png"%i)
    u_old = u.copy()


