import numpy as np
from Code.util import *

N = 8 # How many trails you want
T = 100000

for i in range(N):
    x = trajectory3d(T=T, vi_max=0.08)
    np.save('data/x/%i.npy' % i, x)
