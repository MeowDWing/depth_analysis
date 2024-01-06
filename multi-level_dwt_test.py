import cv2 as cv
import matplotlib.pyplot as plt
import pywt
import numpy as np


def main():

    img = cv.imread('./depth_image/depth0.jpg')

    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cA, (cP, cV, cD) = pywt.dwt2(img_g, 'haar')

    size = cA.shape
    newP = np.zeros(size)
    newV = newD = newP

    img_idwt = pywt.idwt2((cA, (newP, newV, newD)), 'haar')
    # img_idwt = pywt.idwt2(cA, (cP, cV, cD), 'haar')

    plt.figure('0')
    plt.subplot(221)
    plt.imshow(img_g, cmap='gray')
    plt.title('origin')
    plt.subplot(222)
    plt.imshow(img_idwt, cmap='gray')
    plt.title('out')

    errors = img_idwt - img_g
    plt.subplot(223)
    plt.imshow(errors, cmap='gray')
    plt.title('errors')

    plt.subplot(224)
    newP3 = np.zeros(img_g.shape)
    img_restore = pywt.idwt2((img_g, (newP3, newP3, newP3)), 'haar')
    plt.imshow(img_restore, cmap='gray')
    plt.title('double')

    plt.figure('1')
    l2 = pywt.dwt2(cA, 'haar')
    k = 1
    plt.subplot(231)
    plt.imshow(l2[0], 'gray')
    plt.title(f'{k}')

    for c2 in l2[1]:
        plt.subplot(231+k)
        plt.imshow(c2, 'gray')
        plt.title(f'{k}')
        k += 1

    new2P = np.zeros(l2[0].shape)
    img_idwt2_1 = pywt.idwt2((l2[0], (new2P, new2P, new2P)), 'haar')
    img_idwt2 = pywt.idwt2((img_idwt2_1, (newP, newP, newP)), 'haar')
    errors2 = img_idwt2 - img_g
    plt.subplot(231+4)
    plt.imshow(img_idwt2, 'gray')
    plt.title(f'5')
    plt.subplot(231+5)
    plt.imshow(errors2, 'gray')
    plt.title(f'6')

    plt.figure('2')
    l3 = pywt.dwt2(l2[0], 'haar')
    k = 1
    plt.subplot(231)
    plt.imshow(l3[0], 'gray')
    plt.title(f'{k}')

    for c3 in l3[1]:
        plt.subplot(231+k)
        plt.imshow(c3, 'gray')
        plt.title(f'{k}')
        k += 1

    new3P = np.zeros(l3[0].shape)
    img_idwt3_1 = pywt.idwt2((l3[0], (new3P, new3P, new3P)), 'haar')
    img_idwt3_2 = pywt.idwt2((img_idwt3_1, (new2P, new2P, new2P)), 'haar')
    img_idwt3 = pywt.idwt2((img_idwt3_2, (newP, newP, newP)), 'haar')
    errors3 = img_idwt3 - img_g
    plt.subplot(231+4)
    plt.imshow(img_idwt3, 'gray')
    plt.title(f'5')
    plt.subplot(231+5)
    plt.imshow(errors3, 'gray')
    plt.title(f'6')


if __name__ == '__main__':
    main()
    plt.show()
