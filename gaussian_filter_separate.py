import cv2
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

img_blurred = ti.Vector.field(3, dtype=ti.u8, shape=(1024, 1024))
weights = ti.field(dtype=ti.f32, shape=1024, offset=-512)

img2d = ti.types.ndarray(element_dim=1)


@ti.func
def compute_weights(radius, sigma):
    total = 0.0

    # Not much computation here - serialize the for loop to save two more GPU kernel launch costs
    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        # Drop the normal distribution constant coefficients since we need to normalize later anyway
        val = ti.exp(-0.5 * (i / sigma)**2)
        weights[i] = val
        total += val

    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        weights[i] /= total


@ti.kernel
def gaussian_blur(img: img2d, sigma: ti.f32):
    img_blurred.fill(0)
    n, m = img.shape[0], img.shape[1]

    compute_weights(ti.ceil(sigma * 3, int), sigma)
    blur_radius = ti.ceil(sigma * 3, int)

    for i, j in ti.ndrange(n, m):
        l_begin, l_end = max(0, i - blur_radius), min(n, i + blur_radius + 1)
        total_rgb = tm.vec3(0.0)
        total_weight = 0.0
        for l in range(l_begin, l_end):
            w = weights[i - l]
            total_rgb += img[l, j] * w
            total_weight += w

        img_blurred[i, j] = (total_rgb / total_weight).cast(ti.u8)

    for i, j in ti.ndrange(n, m):
        l_begin, l_end = max(0, j - blur_radius), min(m, j + blur_radius + 1)
        total_rgb = tm.vec3(0.0)
        total_weight = 0.0
        for l in range(l_begin, l_end):
            w = weights[j - l]
            total_rgb += img_blurred[i, l] * w
            total_weight += w

        img[i, j] = (total_rgb / total_weight).cast(ti.u8)


img = cv2.imread('images/mountain.jpg')
cv2.imshow('input', img)
gaussian_blur(img, 10)
cv2.imshow('blurred', img)
cv2.waitKey()
