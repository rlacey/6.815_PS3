import numpy as np
from scipy import ndimage
import scipy
from scipy import signal
from numpy import exp

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
  ``` Return a  blured image filtered by box filter```
  out = io.constantIm(im.shape[0], im.shape[1], 0.0)
  shape = np.shape(im)
  xBound = shape[1]
  yBound = shape[0]
  for y,x in imIter(im.copy()):
      val = 0
      counter = 0
    for i in xrange(-k, k):
      for j in xrange(-k, k):
        if not (i is 0 and j is 0):
          val += im[clipY(im, y)][clipX(im, x)]
          counter += 1
    avg = float(val) / counter
    out[x][y] = avg
    val = 0
    counter = 0   
  return out

def convolve(im, kernel):
``` Return an image filtered by kernel```
  

  return im_out

def gradientMagnitude(im):
``` Return the sum of the absolute value of the graident  
    The gradient is the filtered image by Sobel filter ```


  return im_out

def horiGaussKernel(sigma, truncate=3):
  ```Return an one d kernel
  return some_array

def gaussianBlur(im, sigma, truncate=3):

  return gaussian_blured_image


def gauss2D(sigma=2, truncate=3):
  ```Return an 2-D array of gaussian kernel```

  return gaussian_kernel 

def unsharpenMask(im, sigma, truncate, strength):

  return sharpened_image

def bilateral(im, sigmaRange, sigmaDomain):

  return bilateral_filtered_image


def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
  ```6.865 only: filter YUV differently```
  return bilateral_filtered_image

# Helpers

