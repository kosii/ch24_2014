import numpy
import sklearn.cluster
from skimage import io

import math
import pylab
import random

def rrr(cluster_centers):
    return map(lambda l: map(lambda f: int(round(f)), l), cluster_centers)


def transform(cluster_centers, image):
    return [(int(x), int(y), image[int(x), int(y)]) for (x, y, intensity) in cluster_centers]


def generate(w, h, cluster_centers):
    def weight(x, y, center):
        return 1.0/((x-center[0])**2 + (y - center[1])**2)

    def intensity(x, y, cluster_centers):
        for (cx, cy, i) in cluster_centers:
            if x == cx and y == cy:
                return image[x, y]
        s = sum( weight(x, y, center) for center in cluster_centers)
        return sum( weight(x, y, center)*center[2] for center in cluster_centers)/s
    return [[intensity(i, j, cluster_centers) for j in xrange(h)] for i in xrange(w)]


def plot_centers(cluster_centers):
    m = [[0 for j in xrange(h)] for i in xrange(w)]
    for (x, y, intensity) in cluster_centers:
        m[x][y] = intensity
    return m

if __name__ == '__main__':
    # with open()
    filename = "/Users/kosii/Projects/Competitions/CH24/2014/PROBLEMSET/input/J/0.png"
    image = io.imread(filename)
    print image.shape
    (w, h) = image.shape
    dots = []
    for i in xrange(w):
        for j in xrange(h):
            dots.append((i, j, image[i, j]))
    n_clusters = w+h#100
    model = sklearn.cluster.KMeans(n_clusters)#, max_iter=10, n_init=1)
    model.fit(dots)
    print "fitting done"


    pylab.subplot(2, 2, 1)
    pylab.gray()
    print "showing image"
    pylab.imshow(image, interpolation='None')
    pylab.subplot(2, 2, 2)
    pylab.gray()
    # pylab.imshow(numpy.random.rand(16, 16), interpolation='None')
    print "showing generated image"
    pylab.imshow(generate(w, h, transform(model.cluster_centers_, image)), interpolation='None')
    # pylab.imshow(plot_centers(model.cluster_centers_, image), interpolation='None')
    pylab.subplot(2, 2, 3)
    pylab.gray()
    print "showing centers"
    pylab.imshow(plot_centers(transform(model.cluster_centers_, image)), interpolation='None')
    pylab.show()
    # print [random]*10
    # pylab.imshow(rrr(model.cluster_centers_))
    # pylab.show()

