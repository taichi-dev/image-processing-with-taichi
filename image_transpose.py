import cv2
import numpy as np
import taichi as ti

ti.init()

src = cv2.imread("./images/lenna.png")
shape_t = list(src.shape)
shape_t[0], shape_t[1] = shape_t[1], shape_t[0]
dst = np.zeros(shape_t, dtype=src.dtype)


@ti.kernel
def transpose(src: ti.types.ndarray(element_dim=1),
              dst: ti.types.ndarray(element_dim=1)):
    for I in ti.grouped(src):
        dst[I.yx] = src[I]


transpose(src, dst)
cv2.imwrite("lenna_transpose.png", dst)
