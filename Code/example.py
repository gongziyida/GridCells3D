import Hippocampal
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML # Jupyter

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

if __name__ == '__main__':
    ######################### Grid2D #########################
    print('These parameters guarantee the formation of hexagonal grid')
    gridsim = Hippocampal.Grid2D(n=40, m=4, l=0.25, a=1.0, lam=5, c0=1.1, c1=2, tau=10)

    v = np.zeros((5000, 2))
    s, _ = gridsim(alpha=0, T=3000, v=v, print_param=False, spiking=False)
    Hippocampal.plot_sampled_frames(s, title='Initialization')
    plt.show() # Command-line only

    s_init = s[-1].flatten()

    v = np.zeros((8000, 2))
    for i, j in zip(range(8), range(0, 360, 45)):
        theta = j / 180 * np.pi
        v[i*1000:(i+1)*1000] = [np.cos(theta), np.sin(theta)]
    print(v)

    s, spikes = gridsim(alpha=0.2, T=2000, v=v, s_init=s_init, print_param=False, spiking=False)
    ani = Hippocampal.animate({'grid': s}, 2000, title='Counterclockwise motion')
    plt.show() # Command-line only
    # HTML(ani.to_jshtml()) # Jupyter
