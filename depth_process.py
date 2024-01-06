import cv2 as cv
import pywt
import matplotlib.pyplot as plt
import numpy as np


def main():

    images = None
    i = 0
    per = 0.01
    for i in range(1):
        a = cv.imread(f'./depth_image/depth{i}.jpg')
        gray_a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
        # cv.imwrite(f'./depth_image/gray_depth{i}.jpg', gray_a)

        plt.figure(f'right{i}')
        coeff = [0, 0, 0, 0]
        coeff[0], (coeff[1], coeff[2], coeff[3]) = pywt.dwt2(gray_a, 'haar')
        for j in range(4):
            # if (i == 0):
            #     plt.imsave(f'./depth_image/wavelet_{j}.jpg', coeff[j])
            plt.subplot(221+j)
            plt.imshow(coeff[j])
            plt.title(f'{j}')
            # np.savetxt(
            #     f'./depth_image/depth{i}_wavelet{j}_excel.txt', coeff[j], fmt='%d')

        for k in range(25):
            per = 1 - k * 0.04
            # per = -0.1

            plt.figure(f'idwt{i}_per{per}')
            for j in range(4):

                if j > 0:
                    coeff[j] = reserve_only_max(coeff[j], per)
                plt.subplot(231+j)
                plt.imshow(coeff[j])
                plt.imsave(
                    f'./depth_image/depth{i}/coeff{j}_per{per}_img{i}.jpg', coeff[j], cmap='gray')
                plt.title(f'reverse_coe{j}')

            idwt_img = pywt.idwt2(
                (coeff[0], (coeff[1], coeff[2], coeff[3])), 'haar')
            plt.imsave(
                f'./depth_image/depth{i}/reserve_img{i}_per{per}.jpg', idwt_img, cmap='gray')
            plt.subplot(235)
            plt.imshow(idwt_img, cmap='Greys')
            plt.title('idwt')

            e_img = np.abs(gray_a - idwt_img)
            plt.imsave(
                f'./depth_image/depth{i}/error_map_img{i}_per{per}.jpg', e_img, cmap='gray')

    plt.show()


def reserve_only_max(mat: np.array, percent: float):
    mat_abs = np.abs(mat)
    m = np.max(mat_abs)
    threhold = m * (1-percent)
    new_mat = np.where(mat_abs < threhold, 0, mat)
    return new_mat


if __name__ == '__main__':
    main()
