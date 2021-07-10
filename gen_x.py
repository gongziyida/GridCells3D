import time
import numpy as np
from Code.util import *

N = 8 # How many trails you want
T = 100000

for _ in range(N):
    x = trajectory3d(T=T, vi_max=0.08)
    np.save(time.strftime('data/x/%m%d%H%M.npy', time.localtime()), x)
