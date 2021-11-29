import numpy as np
import matplotlib.pyplot as plt


def bin(arr, no_bins):
    bins = np.linspace(np.min(arr), np.max(arr), no_bins)
    bars = np.zeros(len(bins))
    for i, bin in enumerate(bins):
        for j in arr:
            if bins[i-1] < j <= bins[i]: bars[i] += 1
    return(bins, bars)

def bin2(arr, no_bins):
    bins = np.linspace(np.min(arr), np.max(arr), no_bins)
    bins_len = len(bins)
    bars = np.zeros(bins_len)
    if len(arr) < bins_len: print("too many bins")
    for i in arr:
        for j in range(bins_len - 1):
            if bins[j] <= i < bins[j+1]: bars[j] += 1
        if i == bins[bins_len-1]: bars[bins_len-1] += 1
    return(bins, bars)

test = [1, 2, 2, 2, 2.3, 6, 7, 8, 9, 10]

bins, bars = bin2(test, 10)


print(len(test))
what = list(range(len(test)))
print(what)


