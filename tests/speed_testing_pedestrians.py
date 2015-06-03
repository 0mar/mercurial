__author__ = 'omar'
import time

import numpy as np


n = 100000
pos_array = np.random.rand(n, 2)
vel_array = np.random.rand(n, 2)
dt = 0.1
time1 = time.time()
pos_array += vel_array * dt
time2 = time.time()

time3 = time.time()
for i in range(n):
    pos_array[i] += vel_array[i] * dt
time4 = time.time()

print("First: %e" % (time2 - time1))
print("Second: %e" % (time4 - time3))

