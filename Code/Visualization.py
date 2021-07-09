import numpy as np
import sys
from time import strftime
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

pg.setConfigOptions(background='w', foreground='k',
                    imageAxisOrder='row-major', antialias=True)

######################### Global #########################
quali_colors = [(166,206,227),
                (31,120,180),
                (178,223,138),
                (51,160,44),
                (251,154,153),
                (227,26,28),
                (253,191,111),
                (255,127,0),
                (202,178,214),
                (106,61,154),
                (255,255,153),
                (177,89,40)]
quali_colors = [quali_colors[i] for i in range(1, len(quali_colors), 2)] \
             + [quali_colors[i] for i in range(len(quali_colors)-2, -1, -2)]
quali_colors = np.array(quali_colors) / 255

######################### Matplotlib Functions #########################

def plot_cluster(samples, labels, unique_labels, projection='3d'):
    noise = -1 in unique_labels
    n = int(np.ceil(len(unique_labels) / 8) + int(noise))
    fig, ax = plt.subplots(1, n, subplot_kw={'projection': projection}, figsize=(5 * n, 5))
    if n == 1:
        ax = [ax]
    if noise:
        ax[0].scatter(*samples[labels == -1, :3].T, c='y', s=1)

    for i, l in enumerate(unique_labels):
        if l < 0:
            continue
        k = int(noise) + i // 8
        ax[k].set_xlim([-1, 1])
        ax[k].set_ylim([-1, 1])
        if projection == '3d':
            ax[k].set_zlim([-1, 1])
        ax[k].scatter(*samples[labels == l].T, s=5)

def plot_grid2d(x, aang, th=0.7, r=1):
    ''' Plot 2D grid fields with trajectory
    Parameters
    ----------
    x : np.ndarray
        Must be in [-1, 1] along every axis
    aang : np.ndarray
        Must be normalized
    r : int
        Number of neurons to plot. Plot in index order.
    '''
    fig, ax = plt.subplots(1, r, sharex='all', sharey='all')
    if r == 1:
        ax = [ax]

    for k in range(r):
        c = np.zeros((len(aang), 4))
        c[aang[:, k] >= th] = [1, 0, 0, 1]

        ax[k].plot(*x.T, linewidth=1, alpha=0.5, zorder=1) # Trajectory
        ax[k].scatter(x[:, 0], x[:, 1], c=c, s=1, zorder=2) # Scatter

    return fig, ax

def plot_grid3d(x, aang, projection, th=0.8, r=1, n=None, T=None):
    ''' Plot 3D grid fields colored w.r.t. layers or rotation axes

    Parameters
    ----------
    x : np.ndarray
        Must be in [-1, 1] along every axis
    aang : np.ndarray
        Must be normalized
    projection : str
        Valid values: {'2d', '3d'}
    n : int
        Number of layers
    T : int or array-like
        Change inverval or the indices where changes happen
    r : int
        Number of neurons to plot. Plot in index order.
    '''
    if (n is not None) and (T is not None):
        raise ValueError('Ambiguous: which coloring mode do you want?')
    elif n is not None:
        it = range(n)
        idx_f = lambda x, t: (x[:, 2] > t*2/n-1) & (x[:, 2] < (t+1)*2/n-1)
    elif T is not None:
        try:
            it = range(len(T))
        except TypeError:
            T = list(range(0, len(aang), T))
            it = range(len(T))
        idx_f = lambda x, t: np.arange(T[t], len(aang) if t == len(T)-1 else T[t+1])
    else:
        it = (0,)
        idx_f = lambda x, t: np.arange(len(aang))

    # 2D projection
    if projection == '2d':
        fig, ax = plt.subplots(r, 3, sharex='all', sharey='all')
        if r == 1:
            ax = ax[None, :]

        for k in range(r): # Neurons
            idx_spk = aang[:, k] > th
            # c = np.zeros((len(aang), 4))
            # c[aang[:, k] > th, -1] = 1 # Alpha values

            for i, l in enumerate(((0, 1), (0, 2), (1, 2))): # Planes
                for t in it:
                    idx = idx_f(x, t)
                    # c[idx, :3] = quali_colors[t % len(quali_colors)]
                    # ax[k, i].scatter(*x[idx][:, l].T, c=c[idx], s=2)
                    c = quali_colors[t % len(quali_colors)]
                    ax[k, i].plot(*x[idx][idx_spk[idx]][:, l].T, 'o', c=c, markersize=2)
                ax[0, i].set_title(l)

    # 3D
    elif projection == '3d':
        fig, ax = plt.subplots(r, subplot_kw={'projection': '3d'})
        if r == 1:
            ax = [ax]

        for k in range(r):
            idx_spk = aang[:, k] > th
            # c = np.zeros((len(aang), 4))
            # c[aang[:, k] > th, -1] = 1

            for t in it:
                idx = idx_f(x, t)
                # c[idx, :3] = quali_colors[t % len(quali_colors)]
                # ax[k].scatter(*x[idx].T, c=c[idx], s=2)
                c = quali_colors[t % len(quali_colors)]
                ax[k].plot(*x[idx][idx_spk[idx]].T, 'o', c=c, markersize=2)

    fig.tight_layout()
    return fig, ax


def plot_gsmaps(maps):
    fig, ax = plt.subplots(2, sharex='all', sharey='all')
    for i in range(2):
        im = ax[i].imshow(maps[i][..., 0].T, origin='lower', cmap='inferno')
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='3%', pad=0.05)
        fig.colorbar(im, cax=cax)

        ax[i].set_yticks([0, 25, 50])
        ax[i].set_yticklabels([0, r'$\pi/2$', r'$\pi$'], rotation=90)
    ax[1].set_xlabel('Azimuth')
    ax[1].set_xticks([0, 50, 100])
    ax[1].set_xticklabels([0, r'$\pi$', r'$2\pi$'])

    return fig, ax


def plot_sampled_frames(s, fig_r=5, fig_c=5, title=None, fname=None):
    '''
    Parameters
    ----------
    s : np.ndarray
        An array of shape (T, N) storing the neuron metric over time
    fig_r : int, optional
        Number of rows
    fig_c : int, optional
        Number of cols
    title : str, optional
        Super title
    fname : str, optional
        File name to store
    '''
    T = s.shape[0]

    dt = T / (fig_r * fig_c)
    fig, ax = plt.subplots(fig_r, fig_c, figsize=(8, 8), sharex='all', sharey='all')
    for i, j in product(range(fig_r), range(fig_c)):
        t = int(dt * (i * fig_c + j))
        if len(s[t].shape) == 2:
            ax[i, j].imshow(s[t], origin='lower')
        elif len(s[t].shape) == 1:
            ax[i, j].plot(s[t])
        ax[i, j].set_title(t)

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if fname:
        fig.savefig(fname)


def animate(var, T, labels=None, intvl=1, gridintvl=100, title=None, fname=None):
    '''
    Parameters
    ----------
    var : dict
        Variables to animate:
            'bvc': An array of shape (T, n, n) storing the bvc activities
            'grid': An array of shape (T, n, n) storing the grid activities
            'pc': An array of shape (T, n) storing the pc activities
            'traj': A tuple (map, trajectory)
            'vis' : A list of polar coordinates of the visible borders
    labels : dict
        'bvc_alpha', 'bvc_d'
    gridintvl : int
        Inter-frame interval for grid cells
    title : str, optional
        Super title
    fname : str, optional
        File name to store

    Returns
    -------
        ani : matplotlib.animation.FuncAnimation
            Jupyter in-line animation: `HTML(ani.to_jshtml())` (need `IPython.display.HTML`)
    '''
    frames = np.arange(0, T, intvl)

    fig = plt.figure()

    num_ax = len(var)
    if num_ax > 3:
        r, c = 2, int(round(num_ax / 2))
    else:
        r, c = 1, num_ax

    i = 1
    keys = var.keys()
    if 'grid' in keys:
        ax_grid = fig.add_subplot(r, c, i)
        im_grid = ax_grid.imshow(var['grid'][0], origin='lower')
        i += 1
    if 'bvc' in keys:
        ax_bvc = fig.add_subplot(r, c, i)
        im_bvc = ax_bvc.imshow(var['bvc'][0], origin='upper')
        ax_bvc.set_xticks(np.arange(var['bvc'].shape[2]))
        ax_bvc.set_yticks(np.arange(var['bvc'].shape[1]))
        ax_bvc.set_xticklabels(['%.2f' % _ for _ in labels['bvc_alpha']])
        ax_bvc.set_yticklabels(['%.2f' % _ for _ in labels['bvc_d']], rotation=90)
        i += 1
    if 'pc' in keys:
        ax_pc = fig.add_subplot(r, c, i)
        im_pc = ax_pc.plot(var['pc'][0])[0]
        i += 1
    if 'traj' in keys:
        ax_traj = fig.add_subplot(r, c, i)
        plot_borders(var['traj'][0], ax_traj)
        ax_traj.scatter(*var['traj'][1][0], s=5)
        i += 1
    if 'vis' in keys:
        ax_vis = fig.add_subplot(r, c, i, polar=True)
        plot_visible(var['vis'][0], ax_vis)
        i += 1

    if title:
        fig.suptitle(title)

    def ani_f(i):
        ims = []
        if 'grid' in keys:
            im_grid.set_data(var['grid'][i * gridintvl])
            im_grid.autoscale()
            ims.append(im_grid)
        if 'bvc' in keys:
            im_bvc.set_data(var['bvc'][i])
            im_bvc.autoscale()
            ims.append(im_bvc)
        if 'pc' in keys:
            im_pc.set_ydata(var['pc'][i])
            min_s, max_s = var['pc'][i].min(), var['pc'][i].max()
            pad = max_s - min_s
            pad = pad * 0.05 if pad != 0 else 1
            ax_pc.set_ylim(min_s - pad, max_s + pad)
            ims.append(im_pc)
        if 'traj' in keys:
            ims.append(ax_traj.scatter(*var['traj'][1][i], s=5, c='black'))
        if 'vis' in keys:
            ax_vis.clear()
            ims += plot_visible(var['vis'][i], ax_vis)

        return ims

    fig.set_size_inches(w=c*5, h=r*5)
    fig.tight_layout()

    # blit = True : only redraw the changed parts
    ani = matplotlib.animation.FuncAnimation(fig, ani_f, frames=frames, blit=True)

    if fname:
        ani.save(fname)

    return ani


def plot_borders(borders, ax=None, linewidth=2):
    ax = ax if ax else plt
    for b in borders:
        im = ax.plot(*b.T, linewidth=linewidth)
    return im

def plot_visible(visible, ax=None, linewidth=2):
    ax = ax if ax else plt
    for v in visible:
        im = ax.plot(v[1, :] + np.pi / 2, v[0, :], linewidth=linewidth)
    return im


def plot_helper(func):
    def valid(self, input, i):
        if input is not None:
            widgets = func(self, input)
            for w in widgets:
                self.layout.addWidget(w, i%self.rmax, i//self.rmax) # Row, Col
                i += 1
        return i
    return valid


######################### PyQtGraph Functions #########################

class Environment(QtGui.QMainWindow):
    def __init__(self, freq=0.05, title=None, rmax=2, **kwargs):
        '''
        Parameters
        ----------
        freq : float, optional
        title : str, optional
        bvc : tuple, optional
            (bvc_activity, distance_preference, direction_preference)
            where bvc_activity is of shape
            (T, len(direction_preference), len(distance_preference))
        grid : tuple, optional
            (grid_activity, sampling_interval, grid_alphas)
            where grid_activity is of shape (T, n, n)
            or (T, len(grid_alphas), redundancy, n, n)
        traj : tuple
            (map, trajectory)
        visb : list
            A list of polar coordinates of the visible borders
        '''
        super().__init__()
        # Desired Frequency (Hz) = 1 / self.FREQUENCY
        # USE FOR TIME.SLEEP (s)
        self.FREQUENCY = freq

        # Frequency to update plot (ms)
        # USE FOR TIMER.TIMER (ms)
        self.TIMER_FREQUENCY = int(self.FREQUENCY * 1000)

        if title is None:
            self.setWindowTitle(strftime("%Y%m%d_%H%M%S"))
        else:
            self.setWindowTitle(title)

        self.rmax = rmax

        # Outer grid layout
        self.layout = QtGui.QGridLayout()

        cw = QtGui.QWidget()
        cw.setLayout(self.layout)
        self.setCentralWidget(cw)

        # Since _mk_grid changes rmax, it must be the first
        i = self._mk_grid(kwargs.get('grid'), 0)
        i = self._mk_bvc(kwargs.get('bvc'), i)
        i = self._mk_pc(kwargs.get('pc'), i)
        i = self._mk_traj(kwargs.get('traj'), i)
        i = self._mk_visb(kwargs.get('visb'), i)

        # Count the steps
        self._i = 1

        # Make an update timer
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.plot_updater)

        self._mk_toolbar()


    @plot_helper
    def _mk_traj(self, map_traj):
        # Create Plot Widget
        traj_widget = pg.PlotWidget()
        # Enable/disable plot squeeze (Fixed axis movement)
        traj_widget.plotItem.setMouseEnabled(x=False, y=False)

        # Fixed Range
        traj_widget.setXRange(-1, 1)
        traj_widget.setYRange(-1, 1)
        traj_widget.setTitle('Trajectory')

        map, self.traj_data, self.hd_data = map_traj
        self.traj_plot = pg.PlotCurveItem(pen='k')
        traj_widget.addItem(self.traj_plot)

        # Moving dot
        self.dot = pg.ScatterPlotItem()
        self.dot.setBrush(255,0,0)
        traj_widget.addItem(self.dot)

        # Head direction indicator
        self.hd_ind = pg.PlotCurveItem(pen='r')
        traj_widget.addItem(self.hd_ind)

        if not hasattr(self, 'MAX_STEPS'):
            self.MAX_STEPS = len(self.traj_data)

        # Making static map
        for m in map:
            p = pg.PlotCurveItem(pen='k')
            p.setData(x=m[:, 0], y=m[:, 1])
            traj_widget.addItem(p)

        return [traj_widget]

    @plot_helper
    def _mk_visb(self, visb):
        self.visb_widget = pg.PlotWidget()
        self.visb_widget.plotItem.setMouseEnabled(x=False, y=False)

        self.visb_widget.setXRange(-2, 2)
        self.visb_widget.setYRange(-2, 2)
        self.visb_widget.setTitle('Visible Borders')

        self.visb_data = visb

        if not hasattr(self, 'MAX_STEPS'):
            self.MAX_STEPS = len(self.visb_data)

        return [self.visb_widget]

    @plot_helper
    def _mk_grid(self, grid):
        self.gridintvl, grid_alpha = grid[1:] # Unpack
        self.grid_data = grid[0][list(range(0, len(grid[0]), grid[1]))] # Subsampling
        if len(self.grid_data.shape) == 3: # (T, n, n) -> (T, len(alphas), n, n)
            self.grid_data = self.grid_data[:, None, ...]
        self.num_g_alpha = self.grid_data.shape[1]

        self.grid_plots = [pg.ImageItem(axisOrder='row-major')
                           for _ in range(self.num_g_alpha)]

        grid_widgets = [pg.PlotWidget()
                        for _ in range(self.num_g_alpha)]

        for i, (g, p) in enumerate(zip(grid_widgets, self.grid_plots)):
            g.setTitle('2D Grid Cells [%.2f]' \
                        % grid_alpha[i])
            g.plotItem.setMouseEnabled(x=False, y=False)
            g.addItem(p)

        if not hasattr(self, 'MAX_STEPS'):
            self.MAX_STEPS = len(self.grid_data)

        return grid_widgets

    @plot_helper
    def _mk_bvc(self, bvc):
        bvc_widget = pg.PlotWidget()
        bvc_widget.plotItem.setMouseEnabled(x=False, y=False)
        bvc_widget.setTitle('Egocentric Boundary Vector Cells')

        # Label head map
        xticks = bvc_widget.getAxis('bottom')
        xd = [(i + 0.5, '%.2f' % v) for i, v in enumerate(bvc[2])]
        xticks.setTicks([xd])
        yticks = bvc_widget.getAxis('left')
        yd = [(i + 0.5, '%.2f' % v) for i, v in enumerate(bvc[1])]
        yticks.setTicks([yd])

        self.bvc_data = bvc[0]
        self.bvc_plot = pg.ImageItem(axisOrder='row-major')

        bvc_widget.addItem(self.bvc_plot)
        bvc_widget.getViewBox().invertY()

        if not hasattr(self, 'MAX_STEPS'):
            self.MAX_STEPS = len(self.bvc_data)

        return [bvc_widget]

    @plot_helper
    def _mk_pc(self, pc):
        pc_widget = pg.PlotWidget()
        pc_widget.plotItem.setMouseEnabled(x=False, y=False)
        pc_widget.setTitle('Place Cells')

        self.pc_data = pc

        if len(pc.shape) == 2: # Need reshape
            r = int(np.sqrt(pc.shape[1])) # Assume square
            self.pc_data = pc.reshape(pc.shape[0], r, r)

        self.pc_plot = pg.ImageItem(axisOrder='row-major')

        pc_widget.addItem(self.pc_plot)
        pc_widget.getViewBox().invertY()

        if not hasattr(self, 'MAX_STEPS'):
            self.MAX_STEPS = len(self.pc_data)

        return [pc_widget]

    @plot_helper
    def _mk_hdir(self, hdir):
        hdir_widget = pg.PlotWidget()

        hdir_widget.plotItem.setMouseEnabled(x=False, y=False)
        hdir_widget.setTitle('Head Direction Cells')

        self.hdir_data = hdir
        self.hdir_plot = pg.ImageItem(axisOrder='row-major')
        hdir_widget.addItem(self.hdir_plot)

        if not hasattr(self, 'MAX_STEPS'):
            self.MAX_STEPS = len(self.hdir_data)

        return [hdir_widget]

    def _mk_polar_axis(self):
        p = pg.PlotCurveItem(pen=0.8)
        p.setData(x=[-2, 2], y=[0, 0])
        self.visb_widget.addItem(p)

        p = pg.PlotCurveItem(pen=0.8)
        p.setData(x=[0, 0], y=[2, -2])
        self.visb_widget.addItem(p)

        p = pg.PlotCurveItem(pen=0.8)
        p.setData(x=[-2, 2], y=[-2, 2])
        self.visb_widget.addItem(p)

        p = pg.PlotCurveItem(pen=0.8)
        p.setData(x=[-2, 2], y=[2, -2])
        self.visb_widget.addItem(p)

    def _mk_toolbar(self):
        # Add play and pause
        self.toolbar = self.addToolBar('Play/Pause')
        icn = QtGui.QIcon.fromTheme('media-playback-start')
        self.play = QtGui.QAction(icn, 'Play/Pause', self)
        self.play.triggered.connect(self._play_btn_pressed)
        self.play.setCheckable(True)
        self.toolbar.addAction(self.play)


    def _play_btn_pressed(self):
        if self.play.isChecked():
            self.play.setIcon(QtGui.QIcon.fromTheme('media-playback-pause'))
            self.update_timer.start(self.TIMER_FREQUENCY)
        else:
            self.play.setIcon(QtGui.QIcon.fromTheme('media-playback-start'))
            self.update_timer.stop()


    def plot_updater(self):
        if hasattr(self, 'traj_data'):
            self.traj_plot.setData(x=self.traj_data[:self._i, 0],
                                   y=self.traj_data[:self._i, 1])
            self.dot.setData(pos=self.traj_data[None, self._i-1])

            hd = np.zeros((2, 2))
            hd[0] = self.traj_data[self._i-1]
            hd[1] = hd[0] + [np.cos(self.hd_data[self._i-1]) * 0.1,
                             np.sin(self.hd_data[self._i-1]) * 0.1]
            self.hd_ind.setData(x=hd[:, 0], y=hd[:, 1])

        if hasattr(self, 'visb_data'):
            self.visb_widget.clear()
            self._mk_polar_axis()

            for v in self.visb_data[self._i]:
                r = v[0]
                t = v[1] + np.pi / 2
                p = pg.PlotCurveItem(pen='k')
                p.setData(x=r * np.cos(t), y=r * np.sin(t))
                self.visb_widget.addItem(p)

        if hasattr(self, 'grid_data'):
            plots = getattr(self, 'grid_plots')
            for i, plot in enumerate(plots):
                plot.setImage(getattr(self, 'grid_data')[self._i, i])

        for attr in ('bvc', 'pc', 'hdir'):
            if hasattr(self, attr + '_data'):
                plot = getattr(self, attr + '_plot')
                plot.setImage(getattr(self, attr + '_data')[self._i])

        self._i += 1
        if self._i == self.MAX_STEPS:
            self._i = 0

    def get_layout(self):
        return self.layout


if __name__ == '__main__':
    from CellModels import *
    from Drawing import line, trajectory, upsample

    l = []
    l.append(line([0, 0], [0, 2], num=50) - 1)
    l.append(line([0, 2], [1, 2], num=50) - 1)
    l.append(line([1, 2], [2, 2], num=50) - 1)
    l.append(line([2, 2], [2, 0], num=50) - 1)
    l.append(line([2, 0], [0, 0], num=50) - 1)
    l.append(line([1, 1], [1, 2], num=50) - 1)

    T = 3000

    # Trajectory
    x, v, hd = trajectory(np.pi/3, 0.1, T, l, [-0.5, 0])

    # ############ For Grid2D ############
    # gridintvl = 20
    # v = upsample(v, gridintvl)
    #
    # theta = np.zeros(x.shape[0])
    # theta[1:] = np.arctan2(diff[:, 1], diff[:, 0]) # (-pi, pi]
    # theta[theta == -np.pi] = np.pi
    # theta[0] = theta[1]

    # # Get a set of s_init
    # n = 40
    # r = 2
    #
    # grids = [Grid2D(n=n, m=4, l=0.25, a=1.0, lam=5, c0=1.1, c1=2, tau=10,
    #                 print_param=False)
    #          for _ in range(r*r)]
    # s_init_s = np.zeros((r*r, n, n))
    # for i, g in enumerate(grids):
    #     s_init_s[i] = g(alpha=0, T=T, v=np.zeros((T, 2)),
    #                     print_param=False, spiking=False)[0][-1]
    #
    #
    # T = v.shape[0]
    # alphas = (1, 2, 4)
    # s_s = np.zeros((T, len(alphas), r*r, n, n))
    #
    # for i, alpha in enumerate(alphas):
    #     for j, g in enumerate(grids):
    #         s_s[:, i, j, :] = g(alpha=alpha, T=T, v=v, s_init=s_init_s[j],
    #                             print_param=False, spiking=False)[0]

    n = 30
    alphas = [0.2, 0.4, 0.8, 1.6, 3.2]
    g = Grid2D_NNN(n=n, r=0.8, psy=np.random.rand()*np.pi/3,
                   phi=np.random.rand(), s=0.4)
    s_g = g(v, alphas)

    dist_pref = np.array([0.4, 0.3, 0.2, 0.1])
    dir_pref = np.linspace(np.pi, -np.pi, 8, endpoint=False)
    bs = ABVC2D(dist_pref, dir_pref, 5, 0.1, np.pi/4)
    s_b, vis = bs(x, hd, l)

    pc = Place(9, rank=len(alphas), tau=10, scale=0.5, a=20, b=2.5)
    s_p = pc(s_b, s_g)

    # Create main application window
    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('Clearlooks'))

    # Create scrolling plot
    env = Environment(bvc=(s_b, dist_pref, dir_pref),
                      traj=(l, x, hd), visb=vis,
                      grid=(s_g, 1, alphas),
                      pc=s_p)

    env.show()

    # Start Qt event loop unless running in interactive mode or using pyside
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # QtGui.QApplication.instance().exec_()
        app.exec_()
