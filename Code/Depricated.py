import numpy as np

def find_intersections2d(border_points, x, hd, d_th):
    ''' Find intersection points between the lines starting from the query points
        and the points sampled from the borders. Used in the depricated allocentric BVC

    Parameters
    ----------
    border_points : list of np.ndarray's
        The points sampled from the borders.
        The 1st dimension contains different borders (arrays of sampled points).
    x : np.ndarray
        Queries
    hd : np.ndarray
        Allocentric heading directions corresponding to each query, in range [-pi, pi]

    Returns
    -------
    intersec : list
        Intersection information corresponding to each query.
        Each list element contains
        [border index, point index, distance, subtended angl]
        or None if the query does not have a match.
    '''
    if len(hd.shape) == 1: # For convenience
        hd = hd[:, None]

    # Find the points with minimal angular differences
    min_diff_angl = [[] for _ in range(x.shape[0])]
    for i, b_sub in enumerate(border_points): # Iter. over borders
        # Calculate angls
        diff = b_sub[None, ...] - x[:, None, :]
        angl = np.arctan2(diff[..., 1], diff[..., 0]) # [-pi, pi]

        # Calculate angular difference
        diff_angl = np.abs(angl - hd[:, None, 0])
        wrap = diff_angl > np.pi # Need to wrap
        diff_angl[wrap] = np.pi * 2 - diff_angl[wrap]

        # Find minimal angular difference
        m = diff_angl.min(axis=-1)

        # The points with the minimal ... within the threshold are recorded
        p, q = np.where((diff_angl == m[:, None]) & (diff_angl <= np.pi / 2))

        for pi in p:
            dist = np.linalg.norm(x[pi] - b_sub[q[0]]) # Distance
            if dist > d_th and m[pi] > np.pi / 50:
                continue
            min_diff_angl[pi].append([i, q[0], dist])

    intersec = [None for _ in range(x.shape[0])] # Pre-allocate
    for i, li in enumerate(min_diff_angl):
        if len(li) == 0: # Not considered
            continue
        else:
            li.sort(key=lambda x: x[2]) # In case of parallel border segments
            intersec[i] = li[0] # Take the first seen (nearest) segment

            s = border_points[intersec[i][0]][0] - x[i] # Curr to egment start
            t = border_points[intersec[i][0]][-1] - x[i] # Curr to segment end
            norm_s = np.linalg.norm(s)
            norm_t = np.linalg.norm(t)
            if norm_s != 0 and norm_t != 0:
                s /= norm_s
                t /= norm_t
                intersec[i].append(np.arccos(np.dot(s, t))) # Subtended angl [0, pi]
            else:
                intersec[i].append(np.pi / 2) # On one of the ends of the border

    return intersec



class EBVC2D:
    def __init__(self, d, alpha, beta, sig0, sig_ang, d_th):
        ''' Form a 2D grid of allocentric BVCs

        Parameters
        ----------
        d : np.ndarray
            Preferred distances of BVCs
            Give the unique d, do not use np.meshgrid to generate this. Same for alpha.
        alpha : np.ndarray
            Preferred angls of BVCs relative to the head direction (egocentric)
        beta : float
            Rate at which the field increases in size with distance
        sig0 : float
            Radial width of the field at a distance of 0
        sig_ang : float
            STD of angular preference distribution
        d_th : float
            Threshold of angular difference
        '''
        self.d = d
        self.alpha = alpha

        self.beta = beta
        self.sig0 = sig0
        self.sig_ang = sig_ang
        self.sig_rad = sig0 * (d / beta + 1)

        self.d_th = d_th

    def __call__(self, x, hd, map):
        ''' Allocentric BVC Firing

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
        intersec : list
            See find_intersections2d
        '''
        intersec = find_intersections2d(map, x, hd, self.d_th)

        dist = np.zeros(len(x))
        subtend = np.zeros(len(x))
        for i, li in enumerate(intersec):
            dist[i] = np.nan if li is None else li[2]
            subtend[i] = np.nan if li is None else li[3]

        dist_dist = norm.pdf(dist[:, None], loc=self.d[None, :], scale=self.sig_rad[None, :])

        diff_angl = np.abs(hd[:, None] - self.alpha[None, :])
        wrap = diff_angl > np.pi
        diff_angl[wrap] = np.pi * 2 - diff_angl[wrap]
        angl_dist = norm.pdf(diff_angl, scale=self.sig_ang)

        f = dist_dist * angl_dist * subtend[:, None]
        np.nan_to_num(f, copy=False)
        return f, intersec
