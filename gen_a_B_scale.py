import numpy as np
from Code.util import *
import glob

change_ints = (1e1, 1e2, 1e3, 1e4, 1e5)
B_scales = (2, 6, 10, 14, 18)

for k, xfname in enumerate(glob.glob('data/x/*.npy')[:2]):
    tid = xfname[7:15]
    x = np.load(xfname)
    for change_int in change_ints:
        for B_scale in B_scales:
            T = len(x)
            u, r, phi, b = calc_u(x, change_int=change_int)

            # a = sim(u, r, phi, b, B_rotz=8, change_int=change_int, uncert=False)
            # np.save(time.strftime("data/a/c_%m%d%H%M_", t) + str(int(change_int)), a)

            a = sim(u, r, phi, b, B_rotz=8, B_scale=B_scale, change_int=change_int, kappa=400, uncert=True)
            np.save('data/path_int/a_%d/u_' % B_scale + tid + '_' + str(int(change_int)), a)

