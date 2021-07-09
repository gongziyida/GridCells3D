import time
import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.stats import rv_discrete, pearsonr
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import correlate, find_peaks
from scipy.interpolate import RegularGridInterpolator
from Code.Visualization import plot_cluster

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['text.usetex'] = True


############################# Functions #############################

################## Aux Functions ##################

def fibonacci_sphere(n):
    points = np.zeros((n, 3))
    phi = np.pi * (3 - np.sqrt(5)) # golden angle in radians

    points[:, 2] = np.linspace(-1, 1, num=n, endpoint=True) # z goes from 1 to -1
    radius = np.sqrt(1 - points[:, 2]**2) # radius at z

    theta = phi * np.arange(n) # golden angle increment

    points[:, 0] = np.cos(theta) * radius
    points[:, 1] = np.sin(theta) * radius

    return points

def hat(v):
    n = int((1 + np.sqrt(1 + 8 * len(v))) / 2)
    v_ = np.zeros((n, n))

    k = len(v) - 1
    for i in range(n):
        for j in range(i+1, n):
            p = (-1)**k
            v_[i, j] = -p * v[k]
            v_[j, i] = p * v[k]
            k -= 1
    return v_

def vee(v_):
    n = len(v_)
    d = int((n**2 - n) / 2)
    v = np.zeros(d)

    k = d - 1
    for i in range(n):
        for j in range(i+1, n):
            v[k] = (-1)**k * v_[j, i]
            k -= 1
    return v

def activity(a):
    ''' Tansform complex numbers into real periodic neuronal activities in [0, 1]
    '''
    aang = np.real(a)
    maxs = aang.max(axis=0)[None, :]
    mins = aang.min(axis=0)[None, :]
    aang = (aang - mins) / (maxs - mins)
    aang[np.isnan(aang)] = 0
    return aang

def rot_x(x, azimuth, altitude):
    ''' Rotate the positions with the given azimuth and altitude
        Clockwise (w.r.t. each positive axis) as the positive direction
    '''
    cz, sz = np.cos(azimuth), np.sin(azimuth)
    cl, sl = np.cos(altitude), np.sin(altitude)
    rot1 = np.array([[cl, 0, -sl],
                     [0, 1, 0],
                     [sl, 0, cl]])
    rot2 = np.array([[cz, sz, 0],
                     [-sz, cz, 0],
                     [0, 0, 1]])
    return x @ (rot2 @ rot1).T

def oblique_slice(ac, azimuth, altitude, bins=51):
    # Stack coordinates
    x, y, z = ac.shape[:-1] if len(ac.shape) == 4 else ac.shape
    x, y, z = np.linspace(-1, 1, x), np.linspace(-1, 1, y), np.linspace(-1, 1, z)

    ac_interpolator = RegularGridInterpolator((x, y, z), ac, bounds_error=False)

    x0 = np.linspace(-1, 1, bins)
    plane0 = np.stack(np.meshgrid(x0, x0, indexing='ij'), axis=-1).reshape(-1, 2)
    plane0 = np.hstack((plane0, np.zeros((plane0.shape[0], 1))))

    plane = rot_x(plane0, azimuth=azimuth, altitude=altitude)

    return np.nan_to_num(ac_interpolator(plane)).reshape(bins, bins, -1)

def autocorr(f, th):
    ''' Standardized autocorrelation
    '''
    if len(f.shape) == 3:
        f = f[..., None]
    n = np.product(f.shape[:-1])
    f_ = (f - f.mean(axis=(0, 1, 2))) / f.std(axis=(0, 1, 2))

    acs = []
    for i in range(f.shape[-1]):
        ac = correlate(f_[..., i], f_[..., i], mode='full') / n
        ac[ac < th] = 0
        acs.append(ac)
    return np.stack(acs, axis=-1)


############################# Classes #############################

class vonmisesfisher3d_gen:
    """A Von-Mises-Fisher continuous random variable in R^3
    """
    def __init__(self, n=3000):
        self.x = fibonacci_sphere(n)
        self.x_idx = np.arange(self.x.shape[0])

    def pdf(self, mu, kappa):
        if len(mu.shape) != 1:
            raise ValueError('Multidimensional mu is not supported')
        if kappa < 0:
            raise ValueError('Negative kappa')

        mu = mu / np.linalg.norm(mu)
        expk = np.exp(kappa)
        return kappa * np.exp(self.x @ mu * kappa) / (2 * np.pi * (expk - 1/expk))

    def rv_gen(self, mu, kappa):
        pdf = self.pdf(mu, kappa)
        pdf[pdf < 1e-3] = 0
        pmf = pdf / pdf.sum()
        return rv_discrete(values=(self.x_idx, pmf))

    def rvs(self, mu, kappa, size=1):
        return self.rv_gen(mu, kappa).rvs(size=size)

vonmisesfisher3d = vonmisesfisher3d_gen()


################## Generative Functions ##################

U = np.array([[1, 1, 1, 1],
          [1, np.exp(-2j*np.pi/3), np.exp( 1j*np.pi/3), -1],
          [1, np.exp( 1j*np.pi/3), np.exp(-2j*np.pi/3), -1],
          [1, -1, -1, 1]]) / 2

def init_B(scale=7, rotz=15, plot=False):
    ''' Initialize B, and plot
    '''
    B = np.array([[np.sqrt(8/9), 0, -1/3],
              [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
              [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
              [0, 0, 1]])
    M = expm(np.pi / 180 * rotz * hat([0, 0, 1]))
    B = B @ M.T
    B *= scale

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
        ax[0].scatter(*B[:, [0, 1]].T)
        ax[1].scatter(*B[:, [1, 2]].T)
        ax[2].scatter(*B[:, [0, 2]].T)

    return B

def get_Mu(u, col=2):
    if np.linalg.norm(u) < 1e-4:
        raise ValueError

    I = np.identity(3)
    z = np.array([0, 0, 1])
    if (u == z).all():
        return I[:, :col]
    elif (u == -z).all():
        I[[1, 2], [1, 2]] = -1
        return I[:, :col]
    else:
        rot_ax = np.cross(z, u)
        rot_ax /= np.linalg.norm(rot_ax)
        rot_ang = np.arccos(np.dot(u, z))
        Mu = I * np.cos(rot_ang) + (1 - np.cos(rot_ang)) * np.outer(rot_ax, rot_ax) \
             + hat(rot_ax) * np.sin(rot_ang)
        return Mu[:, :col]


def trajectory(a_range, v_range, T, borders, x_init, th=0.05):
    ''' 2D trajectory with gradual head turning

    Parameters
    ----------
    a_range : float
        Angular speed range (-a_range, a_range)
    v_range : float
        Speed range (0, v_range)
    T : int
    borders : list of np.ndarray
    x_init : array-like
        Must be in a valid place (i.e. not out of scope or hit the borders)
    th : float, optional
        The distance threshold that determines whether the agent hits the borders

    Returns
    -------
    x, v, hd : np.ndarray
    '''
    x = np.zeros((T, 2))
    x[0] = x_init
    hd = np.zeros(T) # [-pi, pi]

    hd_pool = np.linspace(-a_range, a_range, num=10)
    step_pool = np.linspace(0, v_range, num=10)

    for t in range(1, T):
        hit_borders = len(borders)

        while hit_borders > 0:
            if t > 1: # Rotate randomly
                hd[t-1] = np.random.choice(hd_pool) + hd[t-2]
            else: # Head dir rand init
                hd[t-1] = (np.random.rand() * 2 - 1) * np.pi

            step = np.random.choice(step_pool) # Random step size

            x[t] = x[t-1] + step * np.array([np.cos(hd[t-1]), np.sin(hd[t-1])])

            x_int = np.linspace(x[t-1], x[t]) # (k, 2)

            # Check whether the agent hits at least one of the borders
            hit_borders = len(borders)
            for b in borders:
                diff = x_int[:, None] - b[None, ...]
                if not (np.linalg.norm(diff, axis=-1) <= th).any():
                    hit_borders -= 1
                    continue

    v = np.diff(x, axis=0)

    return x, v, hd


def trajectory3d(T, vi_max, x_init=None, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    ''' 3D trajectory with gradual motion

    Parameters
    ----------
    vi_max : float
        Speed range for each axis
    T : int
    x_init : array-like
        Must be in a valid place (i.e. not out of scope or hit the borders)
    '''
    x = np.zeros((T, 3))
    if x_init is not None:
        x[0] = x_init

    for t in range(1, T):
        for i, (lb, ub) in enumerate((xlim, ylim, zlim)):
            if (x[t-1, i] < lb) or (x[t-1, i] > ub):
                print(t, x[t-1])
                raise ValueError('Initial x out of Boundary.')
            lb_ = x[t-1, i] - vi_max if x[t-1, i] - vi_max > lb else lb + 1e-5
            ub_ = x[t-1, i] + vi_max if x[t-1, i] + vi_max < ub else ub
            x[t, i] = lb_ + np.random.rand() * (ub_ - lb_)

    return x

def calc_u(x, u_init=np.array([0, 0, 1]), change_int=1, linear=False, kappa=300, u_=None):
    T = len(x)
    if u_ is not None: # Preset u
        assert len(u_) == T
        u = u_
    else:
        u = np.zeros((T, 3))
        u[0] = u_init.copy()

    r = np.zeros(T)
    phi = np.zeros(T)
    b = np.zeros(T)

    MuT = get_Mu(u[0], 3).T

    for t in range(1, T):
        v = x[t] - x[t-1]

        if linear:
            b[t-1] = np.linalg.norm(v)
            u[t-1] = v / b[t-1]

        else:
            v_ = MuT @ v
            b[t-1] = v_[2]
            r[t-1] = np.linalg.norm(v_[:2])

            if r[t-1] == 0: # No planar displacement
                phi[t-1] = 0
            else:
                phi[t-1] = np.arctan2(v_[1], v_[0])

            if u_ is None:
                if t % change_int != 0:
                    u[t] = u[t-1]
                else:
                    # Note: u[t] is between x[t] and x[t+1]
                    j = vonmisesfisher3d.rvs(u[t-1], kappa)
                    u[t] = np.squeeze(vonmisesfisher3d.x[j])
                    normu = np.linalg.norm(u[t])
                    if normu != 1:
                        assert normu != 0
                        u[t] /= normu # Normalize
                    MuT = get_Mu(u[t], 3).T

    return u, r, phi, b

def sim(u, r, phi, b, change_int=1, uncert=False, kappa=600, B_scale=10, B_rotz=8):
    T = len(u)

    B = init_B(B_scale, rotz=B_rotz)

    a = np.zeros((T, 4), dtype='complex')
    a[0] = np.exp(2j * np.pi * np.array([0, 1/2, 0, 3/4]))

    I = np.identity(4)
    u_ = u[0]

    if uncert:
        u_ = np.squeeze(vonmisesfisher3d.x[vonmisesfisher3d.rvs(u[0], kappa)])
    for t in range(1, T):
        if uncert:
            if t % change_int == 0: # u changes here
                u_ = np.squeeze(vonmisesfisher3d.x[vonmisesfisher3d.rvs(u[t-1], kappa)])
        else:
            u_ = u[t-1]
    #     u_ = u_ * 0.9 + u[t-1] * 0.1
        if (r[t-1] == 0) and (b[t-1] == 0):
            a[t] = a[t-1]
            continue

        Mu = get_Mu(u_)
        Wr = U @ np.diag(1j * B @ Mu @ [np.cos(phi[t-1]), np.sin(phi[t-1])]) @ np.conj(U)
        Wb = U @ np.diag(1j * B @ u_) @ np.conj(U)
    #     W = I + Wr * dr[t-1] + Wr @ Wr * dr[t-1]**2/2 + Wb * db + Wb @ Wb * db**2/2
        W = expm(Wr * r[t-1] + Wb * b[t-1])
        a[t] = W @ a[t-1]
    return activity(a)

################## Analytical Functions ##################

def hdbscan_cluster(samples, min_cluster_size=50, min_samples=1, eps=0.01,
                    method='leaf', ignore_range=0.9, plot=True):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=eps,
                                cluster_selection_method=method)
    clusterer.fit(samples)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)

    to_del = []
    for l in unique_labels[1:]:
        m = samples[labels == l].mean(axis=0)
        for i in range(3):
            if (m[i] > ignore_range) or (m[i] < -ignore_range):
                labels[labels == l] = -1
                to_del.append(l + 1) # To indices
                break
    unique_labels = np.delete(unique_labels, to_del)

    if len(unique_labels) == 1:
        raise RuntimeWarning('Only noise detected')

    if plot:
        plot_cluster(samples, labels, unique_labels)

    return unique_labels, labels

def dbscan_cluster(samples, eps=0.2, min_samples=50, ignore_range=0.9, plot=True):
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(samples)
    labels = clusterer.labels_.copy()
    unique_labels = np.unique(labels)

    to_del = []
    for l in unique_labels[1:]:
        m = samples[labels == l].mean(axis=0)
        for i in range(3):
            if (m[i] > ignore_range) or (m[i] < -ignore_range):
                labels[labels == l] = -1
                to_del.append(l + 1) # To indices
                break
    unique_labels = np.delete(unique_labels, to_del)

    if len(unique_labels) == 1:
        raise RuntimeWarning('Only noise detected')

    if plot:
        plot_cluster(samples, labels, unique_labels)

    return unique_labels, labels

def agglo_cluster(samples, min_cluster_size=15, dist_th=0.45, affinity='euclidean',
                  linkage='single', ignore_range=0.9, plot=True):
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_th,
                                        linkage=linkage)
    clusterer.fit(samples)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)

    to_del = []
    for l in unique_labels:
        m = samples[labels == l].mean(axis=0)
        if (labels == l).sum() < min_cluster_size: # Remove small clusters
            to_del.append(l)
            continue
        for i in range(3):
            if (m[i] > ignore_range) or (m[i] < -ignore_range):
                labels[labels == l] = -1
                to_del.append(l) # To indices
                break
    unique_labels = np.delete(unique_labels, to_del)

    if len(unique_labels) == 0:
        raise RuntimeWarning('Only noise detected')

    if plot:
        plot_cluster(samples, labels, unique_labels)

    return unique_labels, labels

def rate_map(x, aang, precision=25, reshape=False):
    ''' Generate 3d histogram for firing rates
    '''
    d = x.shape[-1]
    xrange = np.linspace(-1, 1, precision, endpoint=True)
    n, n_neurons = len(xrange) - 1, aang.shape[-1]
    r = np.zeros((n, n, n, n_neurons)) if d == 3 else np.zeros((n, n, n_neurons))
    for i0 in range(n):
        regionx = (x[:, 0] > xrange[i0]) & (x[:, 0] <= xrange[i0+1])
        for i1 in range(n):
            regiony = (x[:, 1] > xrange[i1]) & (x[:, 1] <= xrange[i1+1])
            regionxy = regionx & regiony
            if d == 3:
                for i2 in range(n):
                    regionz = (x[:, 2] > xrange[i2]) & (x[:, 2] <= xrange[i2+1])
                    region = regionxy & regionz
                    if np.any(region):
                        r[i0, i1, i2] = aang[region].mean(axis=0)
            elif np.any(region): # 2D
                r[i0, i1] = aang[region].mean(axis=0)

    xs = (xrange[1:] + xrange[:-1]) / 2
    xs = np.meshgrid(xs, xs, xs, indexing='ij') if d == 3 else np.meshgrid(xs, xs, indexing='ij')
    xs = np.stack(xs, axis=-1)
    if reshape:
        xs = xs.reshape(-1, d)
        r = r.reshape(-1, n_neurons)
    return r, xs

def hist3d(x, spikes=None, bins=30, lim=((-1, 1), (-1, 1), (-1, 1)), sigma=0):
    ''' A faster and more generic version of `rate_map`
    '''
    if spikes is None:
        f, xs = np.histogramdd(x, bins=bins, range=lim, density=False)
        f = f / f.sum()
    else:
        if len(spikes.shape) == 1:
            spikes = spikes[:, None]
        f = []
        for i in range(spikes.shape[-1]):
            fi, xs = np.histogramdd(x[spikes[:, i] >= 1], bins=bins, range=lim, density=False)
            if sigma > 0: # PSTH
                fi = gaussian_filter(fi, sigma=sigma, mode='constant')
            f.append(fi)
    return np.stack(f, axis=-1), np.stack(xs, axis=-1)


def dist_centroid(samples, unique_labels, labels, plot=True, num_bins=30):
    ''' Compute the distribution of the distances to the centroid
    '''
    dc = [None for i in range(sum(map(lambda x: len(x) - 1, unique_labels)))]
    shamt = 0
    for k in range(len(unique_labels)):
        for i, l in enumerate(unique_labels[k][1:]):
            sel = samples[k][labels[k] == l]
            centroid = sel.mean(axis=0)
            dc[shamt] = np.linalg.norm(sel - centroid, axis=-1)
            shamt += 1

    densities = np.zeros((len(dc), num_bins, 2))
    for i, l in enumerate(dc):
        hist, edges = np.histogram(l, num_bins, density=True, range=(0, 0.5))
        densities[i, :, 0] = (edges[1:] + edges[:-1]) / 2
        densities[i, :, 1] = hist

    if plot:
        mean, std = densities[:, :, 1].mean(axis=0), densities[:, :, 1].std(axis=0)
        q75, q25 = np.percentile(densities[:, :, 1], [75 ,25], axis=0)
        plt.fill_between(densities[0, :, 0], q25, q75, alpha=0.3)
        plt.plot(densities[0, :, 0], mean)

    return dc, densities

def range_dim(samples, unique_labels, labels, plot=True):
    ''' Compute the distribution of the distances
    '''
    rd = np.zeros((sum(map(lambda x: len(x) - 1, unique_labels)), 3))
    shamt = 0
    for k in range(len(unique_labels)):
        for i, l in enumerate(unique_labels[k][1:]):
            assert l != -1
            sel = samples[k][labels[k] == l]
            rd[shamt] = sel.max(axis=0) - sel.min(axis=0)
            shamt += 1
    if plot:
        plt.boxplot(rd, showmeans=True, sym='.', widths=0.3, vert=False)
    return rd

def eig_cov(samples, unique_labels, labels, metric='var_explained', plot=True):
    ''' Compute the distribution of the variance explained (metric='var_explained')
        or the eigenvalues themselves (metric='eig') of the covariance matrices
    '''
    ec = np.zeros((sum(map(lambda x: len(x) - 1, unique_labels)), 3))
    shamt = 0
    for k in range(len(unique_labels)):
        for i, l in enumerate(unique_labels[k][1:]):
            sel = samples[k][labels[k] == l]
            sel = (sel - sel.mean(axis=0)[None, :]) / sel.std(axis=0)[None, :]
            ec[shamt] = np.sort(np.linalg.eigvals(np.cov(sel, rowvar=False)))
            if metric == 'var_explained':
                ec[shamt] /= ec[shamt].sum()
            shamt += 1

    if plot:
        plt.boxplot(ec, showmeans=True, sym='.', widths=0.3, vert=False)

    return ec

def spatial_info(p, f):
    p = p[..., None]
    f_ = f / (f * p).sum(axis=(0, 1, 2))
    f_[f_ == 0] = 1 # Result unchanged since lim_{x->0+} log(x)*x = 0 = log(1)*1
    return (np.log2(f_) * p * f_).sum(axis=(0, 1, 2))

def sparsity_idx(p, f):
    p = p[..., None]
    e_f = (f * p).sum(axis=(0, 1, 2))
    e_f2 = (f**2 * p).sum(axis=(0, 1, 2))
    return e_f**2 / e_f2

def si_shuffle(p, x, spikes, bins=30, sigma=1.5, N=50):
    sinfo_sf = np.zeros((N, spikes.shape[-1]))
    sidx_sf = np.zeros((N, spikes.shape[-1]))
    for i in range(N):
        spikes_sf = [np.random.permutation(spikes[:, j]) for j in range(spikes.shape[-1])]
        spikes_sf = np.stack(spikes_sf, axis=-1)
        f_sf, _ = hist3d(x, spikes_sf, bins=bins, sigma=sigma)
        sinfo_sf[i] = spatial_info(p, f_sf)
        sidx_sf[i] = sparsity_idx(p, f_sf)
    return sinfo_sf, sidx_sf


def autocorr_radial(ac, rmax, method='mean'):
    ''' Planar autocorrelation as a function of distance to the center
    '''
    # Creating coordinates
    radius = int(np.floor(len(ac)/2))
    x = np.arange(-radius, radius+1)
    x = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)

    corr = np.zeros(rmax)
    corr[0] = ac[radius, radius] # Center

    for r in range(1, rmax):
        idxi = x[..., 0]**2 + x[..., 1]**2 > (r-1)**2
        idxo = x[..., 0]**2 + x[..., 1]**2 <= r**2
        idx = idxi & idxo

        if method == 'median':
            corr[r] = np.median(ac[idx])
        elif method == 'mean':
            corr[r] = ac[idx].mean()
        elif method == 'max':
            corr[r] = ac[idx].max()
        else: # quantile
            corr[r] = np.quantile(ac[idx], method)

    return corr

def autocorr_radial3d(ac, rmax, method='mean'):
    radius = int(np.floor(len(ac)/2))
    x = np.arange(-radius, radius+1)
    x = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1)

    corr = np.zeros(rmax)
    corr[0] = ac[radius, radius, radius] # Center

    for r in range(1, rmax):
        idxi = x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2 > (r-1)**2
        idxo = x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2 <= r**2
        idx = idxi & idxo

        if method == 'median':
            corr[r] = np.median(ac[idx])
        elif method == 'mean':
            corr[r] = ac[idx].mean()
        elif method == 'mean_comp':
            corr[r] = ac[idx].sum() / np.sqrt(len(ac[idx]))
        elif method == 'max':
            corr[r] = ac[idx].max()
        else: # quantile
            corr[r] = np.quantile(ac[idx], method)
    return corr

def peak(corr, plane, az, al, width=3, rel_height=0.75):
    ''' Find the first and second peaks
    Returns
    -------
    peaks : sequence
        peaks[0] is the radius of the center peak
        peaks[1] consists of the left and right ends of the second peak
    '''
    corr = np.hstack((corr[-1:0:-1], corr)) # Aux
    res = find_peaks(corr, width=width, rel_height=rel_height)
    peaks = res[0]
    assert len(peaks) % 2 == 1

    c = len(peaks) // 2
    p0r = res[1]['right_bases'][c] - peaks[c]

    if len(peaks) == 1:
        return [p0r, []]

    p1l = res[1]['left_bases'][c+1] - peaks[c]
    p1r = res[1]['right_bases'][c+1] - peaks[c]

    return [p0r, [p1l, p1r]]


def gridness(ac, lb, ub):
    '''
    parameters
    ----------
    ac : np.ndarray
        Planar autocorrelation
    lb : int
        Lower bound
    ub : int
        Inclusive upper bound

    Returns
    -------
    hgs : float
        Hexagonal gridness score
    sgs : float
        Square gridness score
    '''
    assert ac.min() >= -1e-10
    radius = int(np.floor(len(ac)/2))
    x = np.arange(-radius, radius+1)
    x = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)

    idxi = x[..., 0]**2 + x[..., 1]**2 > lb**2
    idxo = x[..., 0]**2 + x[..., 1]**2 <= ub**2
    idx = idxi & idxo

    im = ac.copy() # Process as an image
    im[~idx] = 0
    im = im[radius-ub:radius+ub+1, radius-ub:radius+ub+1] # Clip
    im_flat = im.flatten()

    gs = [0, 0]
    for i, a in enumerate((30, 45)):
        gsmin = min(pearsonr(im_flat, rotate(im, a*2, reshape=False).flatten())[0],
                    pearsonr(im_flat, rotate(im, a*4, reshape=False).flatten())[0])

        gsmax = max(pearsonr(im_flat, rotate(im, a*1, reshape=False).flatten())[0],
                    pearsonr(im_flat, rotate(im, a*3, reshape=False).flatten())[0],
                    pearsonr(im_flat, rotate(im, a*5, reshape=False).flatten())[0])

        gs[i] = gsmin - gsmax

    return gs[0], gs[1]


def gridness_map(ac, az_precision=100, al_precision=50):
    assert ac.min() >= 0
    assert (ac.shape[0] == ac.shape[1]) and (ac.shape[2] == ac.shape[1]) \
            and (ac.shape[0] == ac.shape[2])
    if len(ac.shape) == 3:
        ac = ac[..., None]

    d, n = len(ac), ac.shape[-1]

    hgs_map = np.zeros((az_precision, al_precision, n))
    sgs_map = np.zeros((az_precision, al_precision, n))
    azs = np.linspace(0, np.pi * 2, num=az_precision, endpoint=False)
    als = np.linspace(0, np.pi, num=al_precision, endpoint=False)

    for i, az in enumerate(azs):
        for j, al in enumerate(als):
            plane = oblique_slice(ac, az, al)
            plane[plane < 0] = 0
            for k in range(n):
                corr_radial = autocorr_radial(plane[..., k], int(d * 0.3))
                res = peak(corr_radial, plane, az, al)[1]
                if len(res) == 0:
                    continue
                hgs, sgs = gridness(plane, *res)

                hgs_map[i, j, k] = hgs
                sgs_map[i, j, k] = sgs
    return hgs_map, sgs_map


def chi_score(ac):
    assert ac.min() >= 0
    assert (ac.shape[0] == ac.shape[1]) and (ac.shape[2] == ac.shape[1]) \
            and (ac.shape[0] == ac.shape[2])
    if len(ac.shape) == 3:
        ac = ac[..., None]

    d, n = len(ac), ac.shape[-1]

    azs = np.arange(22, 360, 30) * np.pi / 180
    hgs_map = np.zeros((len(azs), 2, n))
    sgs_map = np.zeros((len(azs), 2, n))

    for i, az in enumerate(azs):
        for j, al in enumerate((72 * np.pi / 180, 56 * np.pi / 180)):
            plane = oblique_slice(ac, az, al)
            plane[plane < 0] = 0
            for k in range(n):
                corr_radial = autocorr_radial(plane[..., k], int(d * 0.3))
                res = peak(corr_radial, plane, az, al)[1]
                if len(res) == 0:
                    continue
                hgs, sgs = gridness(plane, *res)

                hgs_map[i, j, k] = hgs
                sgs_map[i, j, k] = sgs
    chi_fcc = np.median(hgs_map[[3, 7, 11], 0], axis=0) \
            + np.median(sgs_map[[1, 5, 9], 1], axis=0)
    chi_hcp = np.median(hgs_map[[1, 5, 9], 0], axis=0) \
            + np.median(sgs_map[[3, 7, 11], 1], axis=0)
    chi_col = np.median(hgs_map[[0, 2, 4, 6, 8, 10], 0], axis=0) \
            + np.median(sgs_map[[0, 2, 4, 6, 8, 10], 1], axis=0)
    return chi_fcc, chi_hcp, chi_col


def gen_hex_layer(center, r, rotz, bbox=(-1, 1)):
    points = [center]
    lb, ub = 0, 1
    for k in range(round(np.sqrt(8) / r)):
        for center in points[lb:ub]:
            for i in range(6):
                d = i * np.pi / 3 + np.pi / 6 + rotz
                pt = center + r * np.array([np.cos(d), np.sin(d)])
                if (pt < bbox[0]).any() or (pt > bbox[1]).any():
                    continue
                if (lb > 0) and (np.linalg.norm(pt[None, :] - points, axis=1) < 1e-3).any():
                    continue
                points.append(pt)
        lb = ub
        ub = len(points)
    return np.array(points)


def hexagonal_structure(struct_type, r, rotz):
    layera = gen_hex_layer(np.zeros(2), r, rotz, bbox=(-1.5, 1.5))
    d = np.array([r / np.sqrt(3) * np.cos(rotz), r / np.sqrt(3) * np.sin(rotz)])
    layerb = gen_hex_layer(d, r, rotz, bbox=(-1.5, 1.5))
    if struct_type == 'fcc':
        layers, h = (layera, -layerb, layerb), np.sqrt(6) / 3 * r
    elif struct_type == 'hcp':
        layers, h = (layera, -layerb), np.sqrt(6) / 3 * r
    elif struct_type == 'col':
        layers, h = (layera,), r/10
    else:
        raise ValueError

    centers = []
    zs = np.arange(-1.2, 1.2, h)
    for i, z in enumerate(zs):
        l = layers[i % len(layers)]
        centers.append(np.hstack((l, np.ones((l.shape[0], 1)) * z)))

    centers = rot_x(np.vstack(centers), np.pi / 180 * rotz, 0)
    return centers
