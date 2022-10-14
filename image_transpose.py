import cv2
import numpy as np
import taichi as ti

ti.init()

src = cv2.imread("./images/cat.jpg")
h, w, c = src.shape
dst = np.zeros((w, h, c), dtype=src.dtype)


@ti.kernel
def transpose(src: ti.types.ndarray(element_dim=1),
              dst: ti.types.ndarray(element_dim=1)):
    for i, j in ti.ndrange(h, w):
        dst[j, i] = src[i, j]


transpose(src, dst)
cv2.imwrite("cat_transpose.jpg", dst)
