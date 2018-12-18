import cv2, random, numpy


def read_images(images, size):

    samples = random.sample(images, size)

    im = []
    gray = []

    for i in samples:
        paths = i.split('/')
        g = '/'.join(paths[:-2]+['gray', paths[-1]])
        im.append(cv2.imread(i).astype('float32'))
        gray.append(cv2.imread(g).astype('float32'))

    return [numpy.asarray(im), numpy.asarray(gray)]
