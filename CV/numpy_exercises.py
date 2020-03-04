import numpy as np
import cv2

def exercise1():
    arr = np.zeros(10)
    print(arr)
def exercise2():
    arr = np.zeros(10)
    arr[4] = 1
    print(arr)
def exercise3():
    arr = np.arange(10,50)
    print(arr)
def exercise4():
    arr = np.arange(1.0,10.0)
    arr = arr.reshape((3,3))
    print(arr)
def exercise5():
    arr = np.arange(1.0,10.0)
    arr = arr.reshape((3,3))
    arr = np.flip(arr,1)
    print(arr)
def exercise6():
    arr = np.arange(1.0,10.0)
    arr = arr.reshape((3,3))
    arr = np.flip(arr,0)
    print(arr)
def exercise7():
    arr = np.identity(3)
    print(arr)
def exercise8():
    arr = np.random.random_sample((3,3))
    print(arr)
def exercise9():
    arr = np.random.randint(0, 100, 10)
    mean = np.mean(arr)
    print(mean)
def exercise10():
    arr = np.ones((10,10))
    arr[1:9,1:9] = 0
    print(arr)
def exercise11():
    arr = np.zeros(shape=(5, 5))
    arr[:] = np.arange(1,6)
    print(arr)
def exercise12():
    arr = np.random.randint(0, 100, 9)
    arr = np.float64(arr)
    arr = arr.reshape((3,3))
    print(arr)
def exercise13():
    arr = np.random.random_sample((5, 5))
    arr -= arr.mean()
    print(arr)
def exercise14():
    arr = np.random.random_sample((5, 5))
    arr[0,:] -= arr[0,:].mean()
    arr[1,:] -= arr[1,:].mean()
    arr[2,:] -= arr[2,:].mean()
    arr[3,:] -= arr[3,:].mean()
    arr[4,:] -= arr[4,:].mean()
    print(arr)
def exercise15():
    arr = np.random.random_sample((5, 5))
    index = (np.abs(arr-0.5)).argmin()
def exercise16():
    arr = np.random.randint(0, 10, 9)
    arr = arr.reshape((3,3))
    count = arr[arr>5].size
    print(count)
def exercise17():
    arr = np.zeros((64,64))
    arr[:] = np.arange(0.0, 1.0, (1/64))
    cv2.imshow("img",arr)

    k = cv2.waitKey(0)
    if k==27:
     cv2.destroyAllWindows()
def exercise18():
    arr = np.zeros((64,64))
    arr[:] = np.arange(1.0, 0.0, -(1/64))
    arr = np.rot90(arr)
    cv2.imshow("img",arr)

    k = cv2.waitKey(0)
    if k==27:
     cv2.destroyAllWindows()
def exercise19():
    arr = np.ones((64,64,3))
    arr[:,:,0] = 0
    cv2.imshow("img",arr)

    k = cv2.waitKey(0)
    if k==27:
     cv2.destroyAllWindows()
def exercise20():
    arr = np.ones((64,64,3))
    arr[0:32,0:32,0] = 0
    arr[32:64,32:64,2] = 0
    cv2.imshow("img",arr)

    k = cv2.waitKey(0)
    if k==27:
     cv2.destroyAllWindows()
def exercise21():
    arr = cv2.imread('image.png',-1)
    np.insert(arr, 1, 0, axis=1)
    print(arr)
    cv2.imshow("img",arr)

    k = cv2.waitKey(0)
    if k==27:
     cv2.destroyAllWindows()
exercise21()