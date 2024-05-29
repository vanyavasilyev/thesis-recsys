import numpy as np


def apk(is_relevant, k):
    denom = min(k, np.sum(is_relevant))
    num = 0
    hits = 0
    for i in range(k):
        if is_relevant[i]:
            hits += 1
            num += hits / (i + 1)
    if denom == 0:
        return 0
    return num / denom
