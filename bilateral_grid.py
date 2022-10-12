import cv2
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, debug=True)

grid = ti.Vector.field(2, dtype=ti.f32, shape=(512, 512, 128))
grid_blurred = ti.Vector.field(2, dtype=ti.f32, shape=(512, 512, 128))
s_s, s_r = 3, 256


@ti.kernel
def bilateral_filter(img: ti.types.ndarray()):
    # Reset the grid
    print('Scattering', img.shape[0], img.shape[1])
    grid.fill(0)
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        lum = int(img[i, j])
        grid[i // s_s, j // s_s, lum // s_r] += tm.vec2(lum, 1)

    print('Blurring')
    # Grid processing (blur)
    grid_n, grid_m = (img.shape[0] + s_s - 1) // s_s, (img.shape[1] + s_s -
                                                       1) // s_s
    grid_l = (255 + s_r - 1) // s_r
    blur_radius = 3
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_l):
        samples = 0
        p_begin, p_end = max(0, i - blur_radius), min(grid_n,
                                                      i + blur_radius + 1)
        q_begin, q_end = max(0, j - blur_radius), min(grid_m,
                                                      j + blur_radius + 1)
        total = tm.vec2(0, 0)
        for p, q in ti.ndrange((p_begin, p_end), (q_begin, q_end)):
            total += grid[p, q, k]
            samples += 1

        grid_blurred[i, j, k] = total / samples

    print('Slicing')
    # Slicing
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        lum = int(img[i, j])
        sample = grid_blurred[i // s_s, j // s_s, lum // s_r]
        img[i, j] = ti.u8(sample[0] / sample[1])


img = cv2.imread('images/lenna_bw.png')[:, :, 0].copy()
cv2.imshow('input', img)
bilateral_filter(img)
cv2.imshow('filtered', img)
cv2.waitKey()
