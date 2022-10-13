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


@ti.func
def lerp(x, y, a):
    return x * (1 - a) + y * a


@ti.kernel
def bilinear_interp(src: ti.types.ndarray(), dst: ti.types.ndarray()):
    for I in ti.grouped(dst):
        x, y = I.xy / scale
        x1, y1 = int(x), int(y)  # Top-left corner coordinates
        x2, y2 = min(x + 1,
                     h - 1), min(y + 1,
                                 w - 1)  # Bottom-right corner coordinates
        R1 = lerp(src[tm.ivec3(x1, y1, I.z)], src[tm.ivec3(x1, y2, I.z)],
                  y - y1)  # Top horizontal interp
        R2 = lerp(src[tm.ivec3(x2, y1, I.z)], src[tm.ivec3(x2, y2, I.z)],
                  y - y1)  # Bottom horizontal interp
        dst[I] = ti.u8(ti.round(lerp(R1, R2, x - x1)))  # Vertical interp


bilinear_interp(src, dst)
cv2.imwrite("bilinear_interpolation.png", dst)
