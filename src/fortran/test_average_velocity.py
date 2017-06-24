import numpy as np
from velo_averager import average_velocity
import matplotlib.pyplot as plt

# Random positions
n = 100
sx = sy = 10

pos = np.random.random((n, 2))*sx
active = np.ones(n, dtype=bool)
length = sx / n * 100
velos = -(pos - sx / 2)
print(velos)
av_velos = average_velocity(pos,velos,sx,sy,active,length)
print(av_velos)
plt.plot(pos[:,0],pos[:,1],'o')
plt.figure()
plt.quiver(pos[:,0],pos[:,1],velos[:,0],velos[:,1],scale=sx*2)
plt.figure()
plt.quiver(pos[:,0],pos[:,1],av_velos[:,0],av_velos[:,1])
plt.show()