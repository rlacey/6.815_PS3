import numpy as np
##from scipy import ndimage
##import scipy
##from scipy import signal
from numpy import exp
import imageIO as io
from math import e, ceil

############## HELPER FUNCTIONS ###################
def blackImage(im):
    return io.constantIm(im.shape[0], im.shape[1], 0.0)

def imIter(im):
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]):
            yield y, x

def clipX(im, x):
    return min(np.shape(im)[1]-1, max(x, 0))

def clipY(im, y):
    return min(np.shape(im)[0]-1, max(y, 0))

def rgb2yuv(im):    
    matrixYUV = np.array([
                 [0.299000, 0.587000, 0.114000],
                 [-0.14713, -0.28886, 0.436000],
                 [0.615000, -0.51499, -0.10001]
                ])
    return np.dot(im.copy(), matrixYUV.T)


def yuv2rgb(im):
    matrixRGB = np.array([
                 [1, 0.000000, 1.139830],
                 [1, -0.39465, -0.58060],
                 [1, 2.032110, 0.000000]
                ])
    return np.dot(im.copy(), matrixRGB.T)
################# END HELPER ######################

def check_module():
    return 'Ryan Lacey'


def boxBlur(im, k):
    out = blackImage(im)
    mid = int(k/2)
    for y,x in imIter(out):
        for i in xrange(k):
            for j in xrange(k):
                out[y][x] += im[clipY(im, y+i-mid)][clipX(im, x+j-mid)] / (k**2)
    return out


def convolve(im, kernel):
    out = blackImage(im)
    (kRows, kCols) = np.shape(kernel)
    midY = kRows / 2
    midX = kCols / 2    
    for y,x in imIter(im):
        for i in xrange(kRows):
            for j in xrange (kCols):
                out[y][x] += im[clipY(im, y+i-midY)][clipX(im, x+j-midX)] * kernel[i][j]
    return out


def gradientMagnitude(im):
    sobel = np.array([
                     [-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]
                     ])
    horizontal = convolve(im, sobel)
    vertical = convolve(im, sobel.T)
    return (horizontal**2 + vertical**2) ** 0.5


def horiGaussKernel(sigma, truncate=3):
    bound = int(ceil(truncate * sigma))
    kernalLength = 2 * bound + 1
    gaussian = np.array([[e ** (-1 * (position - bound)**2 / (2 * float(sigma)**2)) for position in xrange(kernalLength)]])
    normalized = gaussian / np.sum(gaussian)
    return normalized


def gaussianBlur(im, sigma, truncate=3):
    horizontal = convolve(im, horiGaussKernel(sigma, truncate))
    composite = convolve(horizontal, horiGaussKernel(sigma, truncate).T)
    return composite


def gauss2D(sigma=2, truncate=3):
    return horiGaussKernel(sigma, truncate) * horiGaussKernel(sigma, truncate).T


def unsharpenMask(im, sigma, truncate, strength):
    return im + strength * (im - gaussianBlur(im, sigma, truncate))


def bilateral(im, sigmaRange, sigmaDomain):
    out = blackImage(im)
    truncate = 2
    bound = int(ceil(truncate * sigmaDomain))
    gaussianDistanceKernal = gauss2D(sigmaDomain, truncate)
    for y,x in imIter(im):
        pixelRGB = np.array([0.0, 0.0, 0.0])
        k = 0        
        for i in range(-bound, bound+1):
            for j in range(-bound, bound+1):
                # G(y-y', x-x', o_domain)
                distanceScalar = gaussianDistanceKernal[i+bound][j+bound]
                # I(y,x) - I(y',x')
                pixelValueDifferences = im[y][x] - im[clipY(im, y+i)][clipX(im, x+j)]
                distanceSquared = np.sum(pixelValueDifferences ** 2)
                # G(I(y,x) - I(y',x'), o_range)
                rgbDiffScalar = e ** (-1 * distanceSquared / (2 * sigmaRange**2))
                # G_domain * G_range
                weight = distanceScalar * rgbDiffScalar
                k += weight
                # G_domain * G_range * I_rgb
                pixelRGB += weight * im[clipY(im, y+i)][clipX(im, x+j)]
        out[y][x] = pixelRGB / k
    return out


def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
    out = blackImage(im)
    yuv = rgb2yuv(im)
    bilaY = bilateral(yuv, sigmaRange, sigmaY)
    bilaUV = bilateral(yuv, sigmaRange, sigmaUV)
    for y,x in imIter(im):
        out[y][x] = np.array([bilaY[y][x][0], bilaUV[y][x][1], bilaUV[y][x][2]])
    return yuv2rgb(out)
