import cv2
import numpy as np
import taichi as ti
import taichi.math as tm

ti.init()

src = cv2.imread("./images/lenna_100x100.png")
h, w = src.shape[:2]
scale = 5
shape_s = list(src.shape)
shape_s[0], shape_s[1] = h * scale, w * scale
dst = np.zeros(shape_s, dtype=src.dtype)


@ti.kernel
def bilinear_interp(src: ti.types.ndarray(), dst: ti.types.ndarray()):
    for I in ti.grouped(dst):
        x, y = I.xy / scale
        x1, y1 = int(x), int(y)  # Bottom-left corner
        x2, y2 = min(x + 1, h - 1), min(y + 1, w - 1)  # Top-right corner
        Q11 = src[tm.ivec3(x1, y1, I.z)]
        Q21 = src[tm.ivec3(x2, y1, I.z)]
        Q12 = src[tm.ivec3(x1, y2, I.z)]
        Q22 = src[tm.ivec3(x2, y2, I.z)]
        R1 = tm.mix(Q11, Q21, x - x1)
        R2 = tm.mix(Q12, Q22, x - x1)
        dst[I] = ti.u8(ti.round(tm.mix(R1, R2, y - y1)))


bilinear_interp(src, dst)
cv2.imwrite("bilinear_interpolation.png", dst)
