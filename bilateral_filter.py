import cv2
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, debug=True)

img_filtered = ti.Vector.field(3, dtype=ti.u8, shape=(1024, 1024))


@ti.kernel
def bilateral_filter(img: ti.types.ndarray(element_dim=1), sigma_s: ti.f32,
                     sigma_r: ti.f32):
    img_filtered.fill(0)
    n, m = img.shape[0], img.shape[1]

    blur_radius_s = ti.ceil(sigma_s * 3, int)
    blur_radius_r = ti.ceil(sigma_r * 3, int)

    for i, j in ti.ndrange(n, m):
        k_begin, k_end = max(0,
                             i - blur_radius_s), min(n, i + blur_radius_s + 1)
        l_begin, l_end = max(0,
                             j - blur_radius_s), min(m, j + blur_radius_s + 1)

        total_rgb = tm.vec3(0.0)
        total_weight = 0.0
        for k, l in ti.ndrange((k_begin, k_end), (l_begin, l_end)):
            dist = ((i - k)**2 + (j - l)**2) / sigma_s**2 + (img[i, j].cast(
                ti.f32) - img[k, l].cast(ti.f32)).norm()**2 / sigma_r**2
            # No need to compute Gaussian coeffs here since we normalize in the end anyway
            w = ti.exp(-0.5 * dist)
            total_rgb += img[k, l] * w
            total_weight += w

        img_filtered[i, j] = total_rgb / total_weight

    for i, j in ti.ndrange(n, m):
        img[i, j] = img_filtered[i, j]


img = cv2.imread('images/mountain.jpg')
cv2.imshow('input', img)
bilateral_filter(img, 6, 30)
cv2.imshow('blurred', img)
cv2.waitKey()
