import numpy as np


def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x


def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    if x[~np.isnan(x)].shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)

    return dx


def min_angle_dist(a, b):
    c = a - b
    if np.abs(c) >= np.pi:
        c = c + 2 * np.pi if c < 0 else c - 2 * np.pi
    return c


def interpolate_angle(x):
    x = x.copy()
    for i, a in enumerate(x):
        if np.isnan(a):
            exist_next = False
            for j in range(i, len(x)):
                if not np.isnan(x[j]):
                    exist_next = True
                    break
            if i == 0:
                x[i] = x[j]
                continue
            last = x[i - 1]
            if exist_next:
                next = x[j]
                x[i] = last + (min_angle_dist(next, last)) / (j - i + 1)
                x[i] = (x[i] + np.pi) % (2 * np.pi) - np.pi
            else:
                x[i] = last

    return x