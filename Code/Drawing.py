import numpy as np


def upsample(s, intvl):
    v = np.zeros((s.shape[0] * intvl, s.shape[1]))
    for i in range(s.shape[0]):
        v[i*intvl:(i+1)*intvl] = s[i]
    return v

def line(s, t, num=20, noise=0):
    ''' Generate a line segment

    Parameters
    ----------
    s : array-like
        Coordinates of the start of the segment
    t : array-like
        Coordinates of the end of the segment
    num : int
        Number of points sampled
    noise : float
        Sampling noise strength

    Returns
    -------
    l : np.ndarray
    '''
    l = np.linspace(s, t, num, endpoint=True)
    l += noise * np.random.rand(num, 2)
    return l

def arc(r, s, t, s_c, num=20, noise=0):
    ''' Generate an arc

    Parameters
    ----------
    r : float
        Radius
    s : float
        The starting angl, t >= 0
    t : float
        The terminating angl, t > 0
    s_c : array-like
        Coordinates of the first point
    num : int
        Number of points sampled
    noise : float
        Sampling noise strength

    Returns
    -------
    l : np.ndarray
    '''
    theta = np.linspace(s, t, num, endpoint=True)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    l = np.stack((x, y)).T + noise * np.random.rand(num, 2)
    l -= l[0] + s_c
    return l

def nonlinear(sx, tx, f, num=20, noise=0):
    ''' Generate an nonlinear curve

    Parameters
    ----------
    sx : float
        The starting x
    tx : float
        The terminating x
    f : callable
        Nonlinear function
    num : int
        Number of points sampled
    noise : float
        Sampling noise strength

    Returns
    -------
    l : np.ndarray
    '''
    x = np.linspace(sx, tx, num)
    y = f(x)
    l = np.stack((x, y)).T + noise * np.random.rand(num, 2)
    return l
