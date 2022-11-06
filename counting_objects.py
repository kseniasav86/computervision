from email.mime import image
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.filters.thresholding import (threshold_li, threshold_otsu, threshold_local, threshold_yen)

mask_1 = np.array([[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]])

mask_2 = np.array([[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 1, 1],
                   [1, 1, 0, 0, 1, 1]])

mask_3 = np.array([[1, 1, 0, 0, 1, 1],
                   [1, 1, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]])

mask_4 = np.array([[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [1, 1, 0, 0],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]])

mask_5 = np.array([[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]])

image = np.load("C:\\Users\\USER\\Downloads\\ps.npy.txt")

labeled = label(image)

count = np.max(labeled)
print('Общее количество объектов: ' + str(count))

per1 = binary_erosion(labeled, mask_1)
pr1 = np.max(label(per1))
print("Фигура №1: " + str(pr1))

per2 = binary_erosion(labeled, mask_2)
pr2 = np.max(label(per2))
print("Фигура №2: " + str(pr2 - pr1))

per3 = binary_erosion(labeled, mask_3)
pr3 = np.max(label(per3))
print("Фигура №3: " + str(pr3 - pr1))

per4 = binary_erosion(labeled, mask_4)
pr4 = np.max(label(per4))
print("Фигура №4: " + str(pr4))

per5 = binary_erosion(labeled, mask_5)
pr5 = np.max(label(per5))
print("Фигура №5: " + str(pr5))

plt.subplot(121)
plt.imshow(image)

plt.show()