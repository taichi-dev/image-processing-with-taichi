import cv2
import numpy as np
import taichi as ti
import taichi.math as tm

ti.init()

img = cv2.imread("cameraman.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = np.zeros_like(gray)


def gaussian_filter(src, sigma=1.5, size=5):
    k = size // 2
    gx = ti.Vector([
        ti.exp(-z * z / (2 * sigma * sigma)) /
        ti.sqrt(2 * tm.pi * sigma * sigma) for z in range(-k, k + 1)
    ])
    gaussian_kernel = gx.outer_product(gx)
    gaussian_kernel /= gaussian_kernel.sum()
    h, w = src.shape[:2]
    dst_h = h - size + 1
    dst_w = w - size + 1
    dst = np.zeros((dst_h, dst_w), dtype=src.dtype)

    @ti.kernel
    def smooth(src: ti.types.ndarray(), dst: ti.types.ndarray()):
        h, w = src.shape[:2]
        for i, j in ti.ndrange(dst_h, dst_w):
            cumsum = 0.0
            for ki, kj in ti.static(ti.ndrange(size, size)):
                cumsum += gaussian_kernel[ki, kj] * src[i + ki, j + kj]

            dst[i, j] = ti.u8(ti.round(cumsum))

    smooth(src, dst)
    return dst


def get_gradient(src):
    Gx = ti.Matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = ti.Matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    h, w = src.shape[:2]
    grad = np.zeros(src.shape, dtype=np.uint8)
    dir = np.zeros(src.shape, dtype=np.float32)

    @ti.kernel
    def compute_grad_dir(src: ti.types.ndarray(), grad: ti.types.ndarray(),
                         dir: ti.types.ndarray()):
        for i, j in ti.ndrange(h - 2, w - 2):
            dx = dy = 0.
            for ki, kj in ti.static(ti.ndrange(3, 3)):
                dx += Gx[ki, kj] * src[i + ki, j + kj]
                dy += Gy[ki, kj] * src[i + ki, j + kj]

            grad[i, j] = ti.u8(ti.round(ti.sqrt(dx * dx + dy * dy)))
            if dx <= 1e-4:
                dir[i, j] = tm.pi / 2
            else:
                dir[i, j] = ti.atan2(dy, dx)

    compute_grad_dir(src, grad, dir)
    return grad, dir


def non_maximal_suppression(grad):
    h, w = grad.shape
    nms = np.copy(grad[1:-1, 1:-1])

    @ti.kernel
    def process_nms(grad: ti.types.ndarray(), dir: ti.types.ndarray()):
        for i, j in ti.ndrange((1, h - 1), (1, w - 1)):
            theta = dir[i, j]
            weight = ti.tan(theta)
            if theta > tm.pi / 4:
                d1 = tm.ivec2(0, 1)
                d2 = tm.ivec2(1, 1)
                weight = 1 / weight
            elif theta >= 0:
                d1 = tm.ivec2(1, 0)
                d2 = tm.ivec2(1, 1)
            elif theta >= -tm.pi / 4:
                d1 = tm.ivec2(1, 0)
                d2 = tm.ivec2(1, -1)
                weight = 1 / weight
                weight *= -1
            else:
                d1 = tm.ivec2(0, -1)
                d2 = tm.ivec2(1, -1)
                weight = -1 / weight

            g1 = grad[i + d1.x, j + d1.y]
            g2 = grad[i + d2.x, j + d2.y]
            g3 = grad[i - d1.x, j - d1.y]
            g4 = grad[i - d2.x, j - d2.y]

            f1 = tm.mix(g1, g2, 1 - weight)
            f2 = tm.mix(g3, g4, 1 - weight)
            if f1 > grad[i, j] or f2 > grad[i, j]:
                nms[i - 1, j - 1] = 0

    process_nms()
    return nms()


def double_threshould_filtering(nms, tmin, tmax):
    visited = np.zeros_like(nms)
    dst = np.copy(nms)
    h, w = nms.shape
