import numpy as np
import cv2

def boxFilter(img):
    #normalize for better calculations as float
    img = img/255.0

    #kernel
    #ksize = 11
    #krn = np.zeros((ksize,ksize))
    #krn[:,:]= 1.0/(ksize*ksize)

    krn = gaussianKernel(36)

    #filter
    filtered = convolve(img,krn)

    return filtered

def convolve(img, krn):
    #kernel
    ksize, _ = krn.shape
    krad = int(ksize/2)

    #frame
    height, width, depth = img.shape
    framed = np.ones((height + 2*krad, width + 2*krad, depth))
    framed[krad:-krad, krad:-krad] = img

    #filter
    filtered = np.zeros(img.shape)
    for i in range (0,height):
        for j in range(0, width):
            filtered[i,j] = (framed[i:i+ksize, j:j+ksize]*krn[:,:, np.newaxis]).sum(axis=(0,1))
    
    return filtered

def gaussianKernel(krad):
    sigma = krad/3
    ksize = krad*2 +1
    krn = np.zeros((ksize,ksize))
    for i in range (0, ksize):
        for j in range (0,ksize):
            distance = np.sqrt((krad - i)**2)+((krad - j)**2) #5HEAD
            krn[i,j] = np.exp(-distance**2 / (2*sigma**2))
    return krn/krn.sum()

def bilateralKernel(krad, img, px, py, sigmaColor):
    sigma = krad/3
    ksize = krad*2 + 1
    krn = np.zeros((ksize, ksize))
    for i in range (0, ksize):
        for j in range (0, ksize):
            distance = np.sqrt((krad - i)**2)+((krad - j)**2) #5HEAD   
            po_rgb = img[px][py]
            pn_rgb = img[i][j]
            po_intensity = (po_rgb.max()-po_rgb.min())/po_rgb.max()
            pn_intensity = (pn_rgb.max()-pn_rgb.min())/pn_rgb.max()
            c_range = pn_intensity - po_intensity
            
            krn[i,j] = np.exp(-distance**2 / (2*sigma**2) - (c_range**2) / (2*sigmaColor**2))

    return krn/krn.sum()

def bilateralFilter(img, krad, sigmaColor):
    img = img/255.0
    ksize = krad*2 + 1

    #frame
    height, width, depth = img.shape
    framed = np.ones((height + 2*krad, width + 2*krad, depth))
    framed[krad:-krad, krad:-krad] = img

    filtered = np.zeros(img.shape)
    for i in range (0,height):
        for j in range(0, width):
            krn = bilateralKernel(krad, img, i, j, sigmaColor)
            filtered[i,j] = (framed[i:i+ksize, j:j+ksize]*krn[:,:, np.newaxis]).sum(axis=(0,1))

    return filtered


img = cv2.imread('image.jpg',-1)
img = bilateralFilter(img, 5, 130)
cv2.imshow("MonkaW", img)

k = cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()