import numpy as np


def find_rate(mat: np.array):
    mat = np.abs(mat)
    maxi = np.max(mat)
    sizelist = []
    rate = 1 - 0
    for i in range(100):
        rate = 1 - i/100
        threshold = maxi * rate
        nowm = np.where(mat > threshold, 1, 0)
        size = np.sum(nowm)
        sizelist.append(size)

    diff = []
    for i in range(len(sizelist)-1):
        diff.append(sizelist[i+1]-sizelist[i])

    return sizelist, diff, maxi
