import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, measure)

import os

def pixels_of_cluster(filename):
    image = io.imread(filename, as_gray=True)
    image = image[100:-100]
    image = image[:,200:1050]
    image = np.where(image > 0.18, 0, image)
    return np.count_nonzero(image)

path = "./lab2vids/growth4/"
tsvtitle = "g4.tsv"
frames = sorted(os.listdir(path))

with open(tsvtitle, mode='w') as tsv:
    tsv.write('frame\tarea\n')
    pixels = np.array([])
    i=1
    for frame in frames:
        cluster = pixels_of_cluster(path+frame)
        print('{}\t{}'.format(frame,cluster))
        tsv.write('{}\t{}\n'.format(i,cluster))
        i+=1
        pixels = np.append(pixels, cluster)

fig, ax = plt.subplots()
ax.scatter(range(len(pixels)), pixels, marker='.')
plt.show()
