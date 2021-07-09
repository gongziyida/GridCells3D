import numpy as np
from scipy.stats import norm
from itertools import product


######################### Helper Functions #########################

def A(x, R, dr, a0):
    ''' Input mask for aperiodic grid population model. Used in Grid2D.__Call__

    Parameters
    ----------
    x : np.ndarray
        Cell locations, (N, 2)
    R : float
        Radius of mask that passes input
    dr : float
        Radius of mask that passes input completely
    a0 : float
        Width of edge fading of the mask
    '''
    a = np.ones(x.shape[:-1])

    norm = np.linalg.norm(x, axis=1) # X is of shape (N, 2)
    envlope = R - dr

    if (norm > R).any():
        raise ValueError

    idx = (norm <= R) & (norm >= envlope)
    a[idx] = np.exp(-a0 * ((norm[idx] - envlope) / dr)**2)

    return a


def visible(border_points, x, hd, vl=np.pi/2, vr=np.pi/2):
    ''' Calculate the visible borders

    Parameters
    ----------
    border_points : list of np.ndarray
        The points sampled from the borders, sorted in order.
        The 1st dimension contains different borders (arrays of sampled points).
    x : np.ndarray
        Queries
    hd : np.ndarray
        Head directions, in (-pi, pi]
    vl, vr : float
        Left and right paddings

    Returns
    -------
    out: list of np.ndarray
        distances (axis 0) and angles (axis 1) to visible borders
    '''
    COMPLEMENT = vl + vr > np.pi
    if COMPLEMENT:
        lo, hi = hd + vl, hd - vr
    else:
        hi, lo = hd + vl, hd - vr
    hi[hi > np.pi] -= 2 * np.pi
    lo[lo < -np.pi] += 2 * np.pi

    # Find the angls and distance to each points in the scope
    out = [[] for _ in range(x.shape[0])]
    for i in range(x.shape[0]):
        for b in border_points: # Iter. over borders
            # Calculate angls
            diff = b[None, ...] - x[i, None, :]
            angl = np.arctan2(diff[..., 1], diff[..., 0]) # [-pi, pi]
            angl[angl == -np.pi] = np.pi

            if hi[i] < 0 and lo[i] > 0:
                mask = (angl < hi[i]) | (angl > lo[i])
            else:
                mask = (angl < hi[i]) & (angl > lo[i])

            if COMPLEMENT:
                mask = ~mask

            if mask.sum() == 0:
                out[i].append(None)
                continue

            dist = np.linalg.norm(diff[mask, :], axis=-1)
            angl = angl[mask]
            aux = np.stack((dist, angl))
            out[i].append(aux)

        # Check for overlaps
        for j, b1 in enumerate(out[i]):
            for k, b2 in enumerate(out[i][j+1:], j+1):
                if b1 is None or b2 is None:
                    continue

                bs = [b1[1].copy(), b2[1].copy()] # For convenience
                ends = []

                # Preprocess
                for p, b in enumerate(bs):
                    br = max(b[0], b[-1]) - min(b[0], b[-1])
                    if br > np.pi: # Add one period
                        b[b < 0] += 2 * np.pi
                        bs[p] = b

                    b_end = max(b[0], b[-1])
                    b_start = min(b[0], b[-1])
                    ends += [(p, b_start), (p, b_end)]

                ends.sort(key=lambda x: x[1]) # Ascending
                o1, o2 = ends[1][1], ends[2][1] # Overlapping part; Ascending

                if ends[0][0] == ends[1][0] or abs(o1 - o2) < 1e-4: # No overlap
                    continue

                avg_d1 = b1[0, (bs[0] >= o1) & (bs[0] <= o2)].mean()
                avg_d2 = b2[0, (bs[1] >= o1) & (bs[1] <= o2)].mean()

                if avg_d1 > avg_d2: # b1 is further, rm its overlapping part
                    aux = b1[:, (bs[0] < o1) | (bs[0] > o2)]
                    out[i][j] = aux if aux.shape[1] > 0 else None
                else: # b2 is further, rm its overlapping part
                    aux = b2[:, (bs[1] < o1) | (bs[1] > o2)]
                    out[i][k] = aux if aux.shape[1] > 0 else None

        out[i] = [b for b in out[i] if b is not None] # Clear None

        for j, o in enumerate(out[i]): # Reorient
            o[1] -= hd[i]
            o[1, o[1] < -np.pi] += 2 * np.pi
            o[1, o[1] > np.pi] -= 2 * np.pi
            out[i][j][1] = o[1]

    return out


######################### Cell Classes #########################

class Grid2D:
    def __init__(self, n, m, tau, l, a, lam, c0, c1=1, periodic=True, print_param=True):
        """
        Parameters
        ----------
        n : int
            Number of neurons along either side of the square sheet of grid cells
        m : int
            Number of preferred directions along either side of a square subregion
        tau : float
            Time constant
        l : float
            Shift amount of the DoG center
        a : float
            Amplitude of the excitatory projection
        lam : float
            Approximate distance between two vertices in the formed pattern
        c0 : float
            Ratio between the inhibitory projection width to excitatory projection width
        c1 : float, optional
            Scale of the weights
        periodic : bool, optional
            Periodic network (True) or aperiodic network (False)
        print_param : bool, optional
        """

        if c0 <= 1:
            raise ValueError('Cannot form attractor manifold: c0 <= 1')

        self.N, self.M = n * n, m * m
        self.n, self.m = n, m
        self.tau = tau
        self.l = l
        self.a, self.lam, self.c0, self.c1 = a, lam, c0, c1
        self.beta_i = 1/lam**2
        self.beta_e = self.beta_i * c0

        self.periodic = periodic

        if print_param:
            print('n = %d' % n,
            'm = %d' % m,
            'tau = %.2f' % tau,
            'l = %.2f' % l,
            'a = %.2f' % a,
            'lam = %.2f' % lam,
            'c0 = %.2f' % c0,
            'beta_e = %.2f' % self.beta_e,
            'beta_i = %.2f' % self.beta_i,
            'c1 = %.2f' % c1,
            'periodic %s' % periodic, sep='\n')

        # preferred directions
        self.theta = np.linspace(0, 2 * np.pi, num=self.M, endpoint=False)
        e = np.zeros((n, n))
        for i in range(n//m):
            for j in range(n//m):
                np.random.shuffle(self.theta)
                e[i*m:(i+1)*m, j*m:(j+1)*m] = self.theta.reshape((m, m))
        # unit vector corresponding to the directions
        self.e = np.stack((np.cos(e), np.sin(e)), axis=-1).reshape((self.N, 2))

        # Neuron locations
        _ = np.linspace(-n/2, n/2, num=n, endpoint=False)
        self.x = np.stack(np.meshgrid(_, _), axis=2).reshape((self.N, 2))

        # Calculate distances
        if not periodic:
            loc = self.x - l * self.e
            dist = np.linalg.norm(loc[:, None, :] - self.x[None, ...], axis=-1)**2

        else:
            loc = self.x - l * self.e
            dist = np.abs(loc[:, None, :] - self.x[None, ...])
            dist[dist > n/2] = n - dist[dist > n/2]
            dist = np.linalg.norm(dist, axis=-1)**2

        # Weights from DoG
        self.W = c1 * (a * np.exp(-self.beta_e * dist) - np.exp(-self.beta_i * dist))

    def __call__(self, alpha, T, v, landmark=None,
                 R=None, dr=None, a0=20,
                 s_init=None, spiking=False, print_param=True):
        """
        Parameters
        ----------
        alpha : float
            Velocity input scale
        T : int
            Simulation length
        v : np.ndarray
            Velocity input, with shape (T, 2)
        landmark : np.ndarray, optional
            Landmark neuron input, with shape (T, 1) if given
        R : float, optional
            Radius of mask that passes input
        dr : float, optional
            Radius of mask that passes input completely
        a0 : float, optional
            Width of edge fading of the mask
        s_init : np.ndarray, optional
            Initial neuron state
        spiking : bool, optional
            If the neurons are Poisson spiking neurons
        print_param : bool, optional

        Returns
        -------
        s : np.ndarray
            An array of shape (T, N) storing the neuron states over time
        spike : None or np.ndarray
            An array of shape (T, N) storing the spike train if spiking is enabled
        """

        if R is None:
            R = self.n
        if dr is None:
            dr = R

        if print_param:
            print('alpha = %.2f' % alpha,
                  'T = %d' % T,
                  'R = %d' % R,
                  'dr = %d' % dr,
                  'a0 = %.2f' % a0, sep='\n')

        s = np.zeros((T, self.N))
        # Init states
        if s_init is None: # Very small perturbations
            s[0] = 0 #np.random.rand(self.N) * 0.01
        else:
            s[0] = s_init.flatten()

        spike = np.zeros((T, self.N)) if spiking else None

        # Calculate external input
        b = np.zeros(self.N)
        c = 0.05 if spiking else 1

        for t in range(1, T):
            mask = 1 if self.periodic else A(self.x, R, dr, self.periodic, a0)
            b = 1 + alpha * self.e @ v[t]
            if landmark:
                b += landmark[t]
            b *= mask

            p = c * self.W @ s[t-1] + b # Aux
            p[p < 0] = 0 # ReLU

            pmax = p.max()
            if pmax > 0:
                p /= pmax

            if spiking: # numpy array p is the Poisson probability
                rnd = np.random.rand(self.N)

                spike[t] = (rnd <= p).astype(np.int32)

                s[t, rnd <= p] = s[t-1, rnd <= p] + 1

                ds = -s[t-1, rnd > p]
                s[t, rnd > p] = s[t-1, rnd > p] + ds / self.tau

            else: # p is the synaptic input
                ds = -s[t-1] + p
                s[t] = s[t-1] + ds / self.tau

            s[t, s[t] < 1e-9] = 0 # Avoid Numerical Error

        s = s.reshape((T, self.n, self.n))

        if spiking:
            return spike.reshape((T, self.n, self.n)), s
        else:
            return s, spike


class EBVC2D:
    def __init__(self, d, alpha, beta, sig0, sig_ang):
        ''' Form a 2D grid of egocentric BVCs

        Parameters
        ----------
        d : np.ndarray
            Preferred distances of BVCs
            Give the unique d, do not use np.meshgrid to generate this. Same for alpha.
        alpha : np.ndarray
            Preferred angles of BVCs relative to the head direction (egocentric)
            Assume counterclockwise, i.e. left > 0, right < 0
        beta : float
            Rate at which the field increases in size with distance
        sig0 : float
            Radial width of the field at a distance of 0
        sig_ang : float
            STD of angular preference distribution
        '''
        self.d = d
        self.alpha = alpha

        self.beta = beta
        self.sig0 = sig0
        self.sig_ang = sig_ang
        self.sig_rad = sig0 * (d / beta + 1)

    def __call__(self, x, hd, map):
        ''' EBVC Firing

        Parameters
        ----------
        x : np.ndarray
            Queries
        hd : np.ndarray
            Allocentric heading directions corresponding to each query, in range [-pi, pi]
        map : list of np.ndarray
            The points sampled from the borders.
            The 1st dimension contains different borders (arrays of sampled points).

        Returns
        -------
        f : np.ndarray
            EBVC firing, (T, d, alpha)
        v : list of np.ndarray
            Visible borders
        '''
        v = visible(map, x, hd)
        T = len(v)
        f = np.zeros((T, len(self.d), len(self.alpha)))

        for t in range(T):
            vt_join = np.hstack(v[t])

            dist_respon = norm.pdf(vt_join[0, :, None],
                                   loc=self.d[None, :], scale=self.sig_rad[None, :])

            alpha = self.alpha[None, :] # + hd[t]
            # alpha[alpha > np.pi] -= 2 * np.pi
            # alpha[alpha < -np.pi] += 2 * np.pi
            diff_angl = np.abs(vt_join[1, :, None] - alpha)
            wrap = diff_angl > np.pi
            diff_angl[wrap] = np.pi * 2 - diff_angl[wrap]
            angl_respon = norm.pdf(diff_angl, scale=self.sig_ang)

            f[t] = (dist_respon[..., None] * angl_respon[:, None, :]).sum(axis=0)

        return f, v


class Place:
    def __init__(self, k):
        """
        Parameters
        ----------
        k : int
            Number of place cells
        print_param : bool, optional
        """
        if print_param:
            print('k = %d' % k,
                  sep='\n')
