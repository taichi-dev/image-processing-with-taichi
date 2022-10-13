import cv2
import numpy as np
import taichi as ti

ti.init()

src = cv2.imread("./images/lenna.png")
shape_t = list(src.shape)
shape_t[0], shape_t[1] = shape_t[1], shape_t[0]
dst = np.zeros(shape_t, dtype=src.dtype)


@ti.kernel
def transpose(src: ti.types.ndarray(), dst: ti.types.ndarray()):
    for I in ti.grouped(src):
        J = I
        J[0], J[1] = J[1], J[0]
        dst[J] = src[I]


transpose(src, dst)
cv2.imwrite("lenna_transpose.png", dst)
