import numpy as np
from Code.util import *
import glob

change_ints = (1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5)

for k, xfname in enumerate(glob.glob('data/x/*.npy')):
    tid = xfname[7]
    print(tid)
    x = np.load(xfname)
    for change_int in change_ints:
        T = len(x)
        u, r, phi, b = calc_u(x, change_int=change_int) # Do not need to recalc.

        for kappa in ('c', 600, 500, 400, 300):
            print('\t', kappa, change_int)
            
            if kappa == 'c':
                a = sim(u, r, phi, b, B_rotz=8, change_int=change_int, uncert=False)
                np.save('data/a_c/c_' + tid + '_' + str(int(change_int)), a)
            else:
                a = sim(u, r, phi, b, B_rotz=8, change_int=change_int, kappa=kappa, uncert=True)
                np.save('data/a_%d/u_' % kappa + tid + '_' + str(int(change_int)), a)
        print()
