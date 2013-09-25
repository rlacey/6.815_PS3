import numpy as np
##from scipy import ndimage
##import scipy
##from scipy import signal
from numpy import exp
import imageIO as io
from math import e

############## HELPER FUNCTIONS ###################
def imIter(im):
 for y in xrange(im.shape[0]):
    for x in xrange(im.shape[1]):
       yield y, x

def getBlackPadded(im, y, x):
 if (x<0) or (x>=im.shape[1]) or (y<0) or (y>= im.shape[0]): 
    return np.array([0, 0, 0])
 else:
    return im[y, x]

def clipX(im, x):
   return min(np.shape(im)[1]-1, max(x, 0))

def clipY(im, y):
   return min(np.shape(im)[0]-1, max(y, 0))
              
def getSafePix(im, y, x):
 return im[clipY(im, y), clipX(im, x)]

def point(x, y):
   return np.array([x, y])
              
################# END HELPER ######################

def check_module():
  return 'Ryan Lacey'

def boxBlur(im, k):
  out = io.constantIm(im.shape[0], im.shape[1], 0.0)
  mid = k/2
  for y,x in imIter(out):
      for i in xrange(k):
        for j in xrange(k):
          out[y][x] += im[clipY(im, y+i-mid)][clipX(im, x+j-mid)] / (k**2)
  return out

def convolve(im, kernel):
  (kRows, kCols) = np.shape(kernel)
  midY = kRows / 2
  midX = kCols / 2
  out = io.constantIm(im.shape[0], im.shape[1], 0.0)
  for y,x in imIter(out):
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
  bound = truncate * sigma
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

##def bilateral(im, sigmaRange, sigmaDomain):
##
##  return bilateral_filtered_image
##
##
##def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
##  ```6.865 only: filter YUV differently```
##  return bilateral_filtered_image

# Helpers

