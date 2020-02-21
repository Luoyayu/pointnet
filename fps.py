# farthest Point Sampling python implement
import numpy as np
from matplotlib import pylab as plt

num_points = 100
K = 10  # centroids elected
dim = 2
radius = 25

distance = lambda x1_, x2_: np.sum((x1_ - x2_) ** 2)
del_point = lambda data_, idx_: np.delete(data_, idx_, 0)


def pick_farthest(cs, ps):
    """
    :param cs: selected centroids
    :param ps: remains points
    :return: centroids and it's idx
    """
    dis2set = [.0 for _ in ps]
    for i, p in enumerate(ps):
        dis2set[i] = np.inf
        for j, c in enumerate(cs):
            dis2set[i] = min(dis2set[i], distance(p, c))

    max_dis, max_idx = -np.inf, -1
    for i, p in enumerate(dis2set):
        if p > max_dis:
            max_dis = p
            max_idx = i
    return ps[max_idx], max_idx


def fps(Points, axes):
    Cs = []

    # pick s0 randomly
    s0_idx = np.random.randint(num_points)
    s0 = Points[s0_idx, :]
    Cs.append(s0)

    Points = del_point(Points, s0_idx)

    while len(Cs) < K:
        s, s_idx = pick_farthest(Cs, Points)
        Cs.append(s)
        Points = del_point(Points, s_idx)
    Cs = np.array(Cs)

    plot(axes[1], Cs, Points, "fps sampling")


def rs(Points, axes):
    Cs = []

    while len(Cs) < K:
        s_idx = np.random.randint(num_points - len(Cs))
        s = Points[s_idx, :]
        Points = del_point(Points, s_idx)
        Cs.append(s)

    plot(axes[0], Cs, Points, "randomly sampling")


def plot(ax, cs, points, title: str):
    cs = np.array(cs)
    ax.set_title(title)
    ax.scatter(points[:, 0], points[:, 1], marker='o', alpha=0.6, c='red', label='data points')
    ax.scatter(cs[:, 0], cs[:, 1], alpha=0.6, marker='x', c='blue', label='centroids')
    ax.legend()

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for c in cs:
        ax.add_artist(plt.Circle(c, radius, color='yellow', alpha=0.2))


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, sharex='col', sharey='row')
    Points = np.random.randint(num_points, size=(num_points, dim))
    fig.set_size_inches(20, 10)

    fps(Points, axes)
    rs(Points, axes)
    plt.show()
